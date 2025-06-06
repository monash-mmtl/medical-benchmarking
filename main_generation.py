# pip install google-cloud-aiplatform
# python main_generation.py

import os
import time
import json
import jsonlines
import re
import glob
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel
from google import generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
import vertexai
import argparse
from json_repair import repair_json as json_repair
import copy

# Load environment variables from .env file
load_dotenv()

# Set the environment variable for authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

# Define the default path for diagnoses.jsonl relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIAGNOSES_PATH = os.path.join(SCRIPT_DIR, "tables_list", "diagnoses.jsonl")

# Initialize Vertex AI with your project and credentials
try:
    vertexai.init(
        location="us-central1",
        credentials=None  # Will use service account from environment
    )
    print("✅ Vertex AI initialized successfully")
except Exception as e:
    print(f"❌ Error initializing Vertex AI: {str(e)}")
    exit(1)

# Initialize the model
try:
    model = GenerativeModel("gemini-2.5-pro-preview-03-25")
    print("✅ Model initialized successfully")
    
    # Test the model with a simple prompt
    test_response = model.generate_content("Hello, can you respond?")
    print(f"✅ Test response: {test_response.text[:100]}...")
except Exception as e:
    print(f"❌ Error initializing model: {str(e)}")
    exit(1)

# set model ID for reference
model_id = "gemini-2.5-pro-preview-03-25"

# Define this list to run only specific complaints from the script.
# If this list is empty AND no --complaints are given via CLI, all complaints will be processed.
# CLI --complaints argument will override this list.
COMPLAINTS_TO_RUN_FROM_SCRIPT = [
]

#done cases: "weight loss", "headache", "abdominal pain acute in adults"

COMPLAINTS_BANK = [
    "Abdominal pain, acute in adults",
    "Abdominal pain, chronic or recurrent in adults",
    "Abdominal pain, acute in children",
    "Abdominal pain, recurrent in children",
    "Abdominal pain in women",
    "Abdominal swelling (generalised)",
    "Amenorrhoea",
    "Amnesia, total or partial",
    "Antisocial behaviour in adults",
    "Arm and hand pain (excluding fractures)",
    "Arthralgia/arthritis",
    "Arthralgia/arthritis in children",
    "Back pain, thoracic",
    "Back pain, lower",
    "Breast lumps in women",
    "Breast, nipple discharge",
    "Breast pain (mastalgia)",
    "Calf pain",
    "Chest pain in adults",
    "Chest pain in children",
    "Chronic constipation",
    "Confusion, acute in adults",
    "Cough",
    "Cough, chronic in children",
    "Crying and fussing in infants",
    "Deafness and hearing loss",
    "Diarrhoea",
    "Diplopia",
    "Disturbed or agitated patient",
    "Dizziness/vertigo",
    "Dyspepsia",
    "Dysphagia",
    "Dyspnoea, acute and chronic",
    "Dysuria",
    "Ear discharge (otorrhoea)",
    "Ear pain",
    "Epigastric pain",
    "Epistaxis",
    "Erectile dysfunction",
    "Eye, red and tender",
    "Eye, acute and subacute painless loss of vision",
    "Eye, gradual loss of vision",
    "Facial pain",
    "Falls in the elderly",
    "The febrile child",
    "Fever in the returned traveller",
    "Fever that is prolonged",
    "Fits, faints and funny turns",
    "Foot and ankle pain",
    "Haematemesis",
    "Haematuria",
    "Haemoptysis in adults",
    "Hair loss",
    "Halitosis",
    "Hallucinations",
    "Headache",
    "Hip and buttock pain",
    "Hirsutism in women",
    "Hoarseness",
    "Insomnia",
    "Jaundice in adults",
    "Knee pain",
    "Leg and ankle swelling",
    "Leg pain",
    "Leg ulcers",
    "Limp in children",
    "Lymphadenopathy",
    "Menorrhagia",
    "Mouth conditions, bleeding and painful gums",
    "Mouth conditions, sore tongue",
    "Mouth conditions, ulcers",
    "Nail abnormalities",
    "Nasal drip (rhinorrhoea)",
    "Neck lumps",
    "Neck pain and stiffness",
    "Palpitations",
    "Paraesthesia and numbness",
    "Pelvic pain",
    "Pruritus, generalised",
    "Pruritus, localised skin",
    "Pruritus ani",
    "Purpura",
    "Rectal bleeding",
    "Scrotal pain",
    "Shoulder pain",
    "Skin, acute skin eruptions",
    "Skin, pigmented lesions",
    "Skin, vesicular rash",
    "Skin ulcers",
    "Sore throat",
    "The subfertile couple",
    "Tinnitus",
    "Tiredness/chronic fatigue",
    "Tremor",
    "Urinary incontinence",
    "Vaginal discharge",
    "Vomiting",
    "Vulvar discomfort and irritation",
    "Weight gain",
    "Weight loss"
]

def load_differential_diagnoses(jsonl_file_path=None):
    """Load all differential diagnoses from a JSONL file, with category information."""
    if jsonl_file_path is None:
        jsonl_file_path = DEFAULT_DIAGNOSES_PATH
    differentials_by_complaint = {}
    # Also track category information for each diagnosis
    diagnosis_categories_by_complaint = {}
    
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    # Expecting a single key per JSON object, which is the complaint
                    if len(data) == 1:
                        complaint_name = list(data.keys())[0]
                        diagnosis_categories = data[complaint_name]
                        
                        all_diagnoses_for_complaint = []
                        diagnosis_to_category_map = {}  # Maps each diagnosis to its category
                        
                        if isinstance(diagnosis_categories, dict):
                            for category_key, diagnoses_list in diagnosis_categories.items():
                                # Exclude specified categories
                                if category_key in ["Masquerades", "Patient trying to tell me something"]:
                                    continue # Skip these categories
                                    
                                # We are interested in "Probability diagnosis", "Serious disorders", "Pitfalls",
                                # and potentially others if they contain lists of strings (after exclusion).
                                if isinstance(diagnoses_list, list):
                                    for diag in diagnoses_list:
                                        if isinstance(diag, str):
                                            diag_str = str(diag)
                                            all_diagnoses_for_complaint.append(diag_str)
                                            # Store category of each diagnosis
                                            diagnosis_to_category_map[diag_str] = category_key

                            # Remove duplicates (if a diagnosis appears in multiple categories, keep first occurrence)
                            seen = set()
                            unique_diagnoses = []
                            for diag in all_diagnoses_for_complaint:
                                if diag not in seen:
                                    seen.add(diag)
                                    unique_diagnoses.append(diag)
                            
                            if unique_diagnoses:
                                differentials_by_complaint[complaint_name] = unique_diagnoses
                                diagnosis_categories_by_complaint[complaint_name] = diagnosis_to_category_map
                                print(f"✅ Loaded {len(unique_diagnoses)} unique differentials for '{complaint_name}' from {jsonl_file_path}")
                            else:
                                print(f"⚠️ No valid diagnoses found for '{complaint_name}' in {jsonl_file_path} on line {line_number}")
                        else:
                            print(f"❌ Unexpected structure for complaint '{complaint_name}' on line {line_number} in {jsonl_file_path}. Expected dict of categories.")
                    else:
                        print(f"❌ Unexpected JSON structure on line {line_number} in {jsonl_file_path}. Expected single key (complaint name).")
                except json.JSONDecodeError as e:
                    print(f"❌ Error decoding JSON on line {line_number} in {jsonl_file_path}: {str(e)}")
                except Exception as e:
                    print(f"❌ Error processing line {line_number} in {jsonl_file_path}: {str(e)}")
    except FileNotFoundError:
        print(f"❌ Error: The file {jsonl_file_path} was not found.")
        return {}, {} # Return empty if file not found to prevent crash
    except Exception as e:
        print(f"❌ An unexpected error occurred while trying to read {jsonl_file_path}: {str(e)}")
        return {}, {}

    if not differentials_by_complaint:
        print(f"⚠️ No differential diagnoses were loaded from {jsonl_file_path}. Please check the file content and path.")
        
    return differentials_by_complaint, diagnosis_categories_by_complaint

def validate_specific_key_nesting(data_dict, base_path_name="case"):
    """
    Validates if "OSCE_Examination" and its specific child keys exist as expected
    within the provided data_dict (assumed to be the content of the "case" key).

    Expected structure:
    case: {
        "OSCE_Examination": {
            "Patient_Actor": { ... },
            "Physical_Examination_Findings": { ... },
            "Test_Results": { ... },
            "Correct_Diagnosis": "Some diagnosis"
        }
    }
    """
    # 1. Check for OSCE_Examination at the top level of data_dict (which is the content of "case")
    osce_key = "OSCE_Examination"
    osce_path_str = f"{base_path_name}.{osce_key}" # e.g., "case.OSCE_Examination"

    if not isinstance(data_dict, dict):
        # This check is mostly for robustness, as data_dict should be the "case" object.
        return False, f"Validation Error: Expected '{base_path_name}' to be a dictionary, but found {type(data_dict).__name__}."

    if osce_key not in data_dict:
        return False, f"Validation Error: Missing key '{osce_key}' at path '{base_path_name}' (expected '{osce_path_str}')."

    osce_level_dict = data_dict[osce_key]
    if not isinstance(osce_level_dict, dict):
        return False, f"Validation Error: Key '{osce_key}' at path '{osce_path_str}' should point to a dictionary, but found {type(osce_level_dict).__name__}."

    # 2. Define keys expected as direct children of OSCE_Examination and their requirements.
    # Format: (key_name, must_be_dict_if_not_terminal_key)
    # "Correct_Diagnosis" is the terminal key for diagnosis; its value type isn't strictly checked beyond existence here.
    expected_children_of_osce = [
        ("Patient_Actor", True),
        ("Physical_Examination_Findings", True),
        ("Test_Results", True),
        ("Correct_Diagnosis", False) # False means it doesn't *have* to be a dict for this validation.
    ]

    # The base path for error messages concerning children of OSCE_Examination
    children_base_path_str = osce_path_str # This will be like "case.OSCE_Examination"

    for child_key, must_be_dict in expected_children_of_osce:
        current_child_path_str = f"{children_base_path_str}.{child_key}" # e.g., "case.OSCE_Examination.Patient_Actor"

        if child_key not in osce_level_dict:
            return False, f"Validation Error: Missing key '{child_key}' at path '{current_child_path_str}'."

        if must_be_dict:
            child_value = osce_level_dict[child_key]
            if not isinstance(child_value, dict):
                return False, f"Validation Error: Key '{child_key}' at path '{current_child_path_str}' should point to a dictionary, but found {type(child_value).__name__}."
    
    # If all checks pass
    return True, f"Specific key nesting validated successfully ({osce_path_str} and its direct children)."

# Restore the example_json_format variable definition
example_json_format = """
{
      "OSCE_Examination": {
        "Patient_Actor": {
          "Demographics": "32-year-old male",
          "History": "The patient reports a progressive worsening of headaches over the past several weeks. He also notices blurred vision occasionally, especially later in the day. He initially attributed these to work-related stress but has decided to seek medical attention as the symptoms persisted and have not responded to over-the-counter medications.",
          "Symptoms": {
            "Primary_Symptom": "Headaches",
            "Secondary_Symptoms": [
              "Blurred vision",
              "Pain intensifying over the day",
              "Occasional nausea"
            ]
          },
          "Past_Medical_History": "Previous concussion during a motor vehicle accident 5 years ago. No other significant past medical or surgical history.",
          "Social_History": "Non-smoker, occasional drinker. Works as a software engineer.",
          "Review_of_Systems": "Pain is located in the frontal region and worse in the morning. Denies fever, focal neurology, seizures, balance problems, weakness or loss of consciousness. No preceding aura."
        },
        "Physical_Examination_Findings": {
          "Vital_Signs": {
            "Temperature": "36.7°C (98°F)",
            "Blood_Pressure": "128/75 mmHg",
            "Heart_Rate": "60 bpm",
            "Respiratory_Rate": "18 breaths/min"
          },
          "Neurological_Examination": {
            "Pupils": "Bilateral papilledema",
            "Gait": "Normal gait",
            "Motor_System": "Normal power and tone in all four limbs. Deep tendon reflexes within normal limits.",
            "Sensory_System": "Normal sensation to touch, pain, temperature, and vibration."
          }
        },
        "Test_Results": {
          "Imaging": {
            "MRI_Brain": {
              "Findings": "Normal, no space-occupying lesions."
            },
            "CT_Brain": {
              "Findings": "Normal, no space-occupying lesions."
            },
            "CT_Angiogram_Brain": {
              "Findings": "Normal, no acute intracranial hemorrhage."
            }
          },
          "CSF_Opening_Pressure": "Increased CSF opening pressure",
          "CSF_analysis": {
            "Protein_Level": "Normal",
            "WBC_Count": "Normal"
          }
        },
        "Correct_Diagnosis": "Idiopathic intracranial hypertension"
      }
    },
{
    "Presenting complaint": "Weight loss",
    "case": {
      "OSCE_Examination": {
        "Patient_Actor": {
          "Demographics": "60-year-old female",
          "History": "The patient reports a 6-month history of diarrhea, as well as considerable weight loss during this period. She also mentions being easily fatigued and having been diagnosed with iron deficiency anemia.",
          "Symptoms": {
            "Primary_Symptom": "Weight loss and chronic diarrhea",
            "Secondary_Symptoms": [
              "Fatigue",
              "Iron deficiency anemia",
		"Constipation"
            ]
          },
          "Past_Medical_History": "No significant past medical history.",
          "Social_History": "Non-smoker, moderate alcohol consumption, works as a restaurant manager. No allergies, eats an unrestricted diet.",
          "Review_of_Systems": "Patient denies experiencing any vomiting, blood in stool, or gastric pain. Reports 5 kg weight loss over 3 months, denies any rashes."
        },
        "Physical_Examination_Findings": {
          "Vital_Signs": {
            "Temperature": "36.7°C (98.1°F)",
            "Blood_Pressure": "115/70 mmHg",
            "Heart_Rate": "82 bpm",
            "Respiratory_Rate": "17 breaths/min"
          },
          "Abdominal_Examination": {
            "Inspection": "Flat abdomen",
            "Auscultation": "Increased bowel sounds",
            "Percussion": "Tympanic sound throughout",
            "Palpation": "Mild, diffuse abdominal tenderness"
          }
        },
        "Test_Results": {
          "Blood_Tests": {
            "Hemoglobin": "Low",
            "Iron_Studies": {
              "Serum_Iron": "Low",
              "Transferrin_Saturation": "Low",
              "Ferritin": "Low",
		"Anti TTG-IgA": "Positive",
		"Anti-EMA": "Positive",
		"IgA-Titre": "Normal"
            }
          },
          "Stool_Analysis": {
            "Macroscopic_Appearance": "Normal",
            "Fecal_Fat_Stain": "Positive"
          },
          "Upper_Endoscopy": {
            "Findings": "Villous atrophy in the duodenum"
          },
          "Small_Bowel_Biopsy": {
            "Histopathology": "Villous atrophy with crypt hyperplasia and intraepithelial lymphocytosis"
          },
"Colonoscopy": {
            "Findings": "Normal"
          }
        },
        "Correct_Diagnosis": "Celiac disease"
      }
    }
  },
{
  "Presenting complaint": "Abdominal pain, migrating from epigastrium to right lower quadrant",
  "case": {
    "OSCE_Examination": {
      "Patient_Actor": {
        "Demographics": "28-year-old male",
        "History": "The patient presents with a 36-hour history of abdominal pain. He states the pain initially began around his epigastrium, described as a persistent dull ache. Over the last 12 hours, the pain has become more constant and has shifted, now predominantly felt in the right lower quadrant, with some associated discomfort radiating to his right flank. He rates the pain as 5/10, exacerbated by movement and coughing. He experienced one episode of nausea this morning but has not vomited. His appetite is noticeably reduced, though he is not completely anorexic. His last bowel movement was yesterday and was normal; he now feels slightly constipated. He also reports a mild burning sensation at the end of urination for the past day, but denies increased urinary frequency, urgency, or visible blood in his urine. He mentions consuming some 'questionable leftovers' two days prior, but no one else who ate them has reported illness.",
        "Symptoms": {
          "Primary_Symptom": "Abdominal pain, migrating from epigastrium to right lower quadrant",
          "Secondary_Symptoms": [
            "Nausea (one episode)",
            "Reduced appetite",
            "Slight constipation",
            "Mild dysuria"
          ]
        },
        "Past_Medical_History": "Generally healthy. No chronic illnesses. No previous abdominal surgeries. No history of inflammatory bowel disease. No known drug allergies.",
        "Social_History": "Non-smoker. Drinks socially, approximately 2-3 beers on weekends. Works as a graphic designer. Denies illicit drug use.",
        "Medications": "No regular medications. Takes ibuprofen occasionally for headaches.",
        "Review_of_Systems": "Patient reports the abdominal pain as his main concern. Denies fever at home (though measured 37.6°C in clinic), denies chills or night sweats. Denies significant recent weight loss. Denies chest pain, palpitations, or shortness of breath. Denies melena or hematochezia. Denies rash or joint pains. Confirms mild dysuria but denies frank hematuria, true urinary urgency, or flank pain distinct from his described abdominal discomfort."
      },
      "Physical_Examination_Findings": {
        "Vital_Signs": {
          "Temperature": "37.6°C (99.7°F)",
          "Blood_Pressure": "125/80 mmHg",
          "Heart_Rate": "92 bpm",
          "Respiratory_Rate": "18 breaths/min",
          "Oxygen_Saturation": "99% on room air"
        },
        "General_Appearance": "Alert and oriented. Appears uncomfortable and is observed to shift position frequently on the examination table. Not in acute distress.",
        "Abdominal_Examination": {
          "Inspection": "Abdomen is flat. No visible scars, distension, or discoloration. Umbilicus is central and inverted.",
          "Auscultation": "Bowel sounds are present but slightly hypoactive in all four quadrants. No abdominal bruits.",
          "Percussion": "Mild tympany throughout. Tender to light percussion over the right lower quadrant and extending slightly to the right flank. No shifting dullness.",
          "Palpation": "Maximal tenderness on palpation in the right lower quadrant, slightly medial to McBurney's point. Some voluntary guarding noted in this area. No definite rebound tenderness elicited, though the patient winces and pulls away upon rapid withdrawal of pressure. Rovsing's sign is equivocal. Psoas and Obturator signs are negative. No palpable masses. Liver edge not palpable; spleen not palpable. No hepatosplenomegaly."
        },
        "Cardiovascular_Examination": "Regular rate and rhythm. S1 and S2 heart sounds normal. No murmurs, rubs, or gallops.",
        "Respiratory_Examination": "Chest clear to auscultation bilaterally. No wheezes, rales, or rhonchi. Symmetric chest expansion.",
        "Back_Examination": "No costovertebral angle tenderness elicited bilaterally."
      },
      "Test_Results": {
        "Blood_Tests": {
          "Complete_Blood_Count": {
            "WBC": "11.5 x 10^9/L",
            "Hemoglobin": "14.5 g/dL",
            "Hematocrit": "43%",
            "Platelets": "250 x 10^9/L",
            "Neutrophils": "75%",
            "Lymphocytes": "18%",
            "Monocytes": "5%",
            "Eosinophils": "2%"
          },
          "C_Reactive_Protein": "35 mg/L",
          "Electrolytes_and_Renal_Function": {
            "Sodium": "138 mmol/L",
            "Potassium": "4.0 mmol/L",
            "Chloride": "102 mmol/L",
            "Bicarbonate": "24 mmol/L",
            "Urea": "5.0 mmol/L",
            "Creatinine": "80 µmol/L"
          },
          "Liver_Function_Tests": {
            "ALT": "25 U/L",
            "AST": "22 U/L",
            "ALP": "70 U/L",
            "Total_Bilirubin": "0.8 mg/dL"
          },
          "Amylase": "60 U/L"
        },
        "Urine_Analysis": {
          "Dipstick": {
            "Color": "Yellow",
            "Clarity": "Clear",
            "Specific_Gravity": "1.015",
            "pH": "6.0",
            "Leukocytes": "Trace",
            "Nitrite": "Negative",
            "Protein": "Negative",
            "Glucose": "Negative",
            "Ketones": "Negative",
            "Urobilinogen": "Normal",
            "Bilirubin": "Negative",
            "Blood": "Trace"
          },
          "Microscopy": {
            "WBC_per_HPF": "2-5",
            "RBC_per_HPF": "1-3",
            "Bacteria": "None seen",
            "Casts": "None seen",
            "Epithelial_cells": "Few"
          }
        },
        "Imaging": {
          "Abdominal_Ultrasound_RLQ_Pelvis": {
            "Findings": "Focused examination of the right lower quadrant demonstrates limited views due to overlying bowel gas. The appendix was not definitively visualized. No free fluid or obvious abscess collection identified in the RLQ. Kidneys appear sonographically normal bilaterally. Bladder unremarkable. No other acute sonographic abnormality identified to explain the patient's symptoms."
          },
          "CT_Abdomen_Pelvis_with_IV_contrast": {
            "Findings": "The appendix is identified, measuring 7mm in maximal transverse diameter with associated mild circumferential wall thickening and periappendiceal fat stranding. A small amount of simple free fluid is noted adjacent to the appendiceal tip. No definite appendicolith visualized. No evidence of abscess formation or frank perforation. Several mildly prominent mesenteric lymph nodes, up to 8mm in short axis, are seen in the right lower quadrant, likely reactive. The visualized small and large bowel are otherwise unremarkable. Liver, spleen, pancreas, kidneys, and adrenal glands are normal. No acute pelvic pathology identified."
          }
        }
      },
      "Correct_Diagnosis": "Acute appendicitis (non-perforated)"
    }
  }
}

"""

# Helper function to sanitize filenames
def sanitize_filename(name):
    """Replace invalid filename characters with underscores."""
    # Replace characters that are invalid in filenames with underscores
    # Windows specifically disallows: \ / : * ? " < > |
    invalid_chars = r'[\\/:*?"<>|]'
    return re.sub(invalid_chars, '_', name)

# Generate cases for each differential diagnosis
def generate_cases_from_differentials(max_cases_per_complaint=0, specific_complaints=None):
    # Load all differentials
    differentials_by_complaint, diagnosis_categories_by_complaint = load_differential_diagnoses()
    
    # Determine which complaints to process based on priority:
    # 1. CLI (--complaints argument)
    # 2. COMPLAINTS_TO_RUN_FROM_SCRIPT (if CLI not used and list is not empty)
    # 3. All complaints (if neither CLI nor script list is used)
    
    complaints_to_process_source = "all available"
    final_complaints_list = None

    if specific_complaints: # CLI argument takes precedence
        final_complaints_list = specific_complaints
        complaints_to_process_source = "command-line argument"
        print(f"ℹ️ Processing complaints specified via command-line: {', '.join(final_complaints_list)}")
    elif COMPLAINTS_TO_RUN_FROM_SCRIPT: # Check script variable if CLI not used
        final_complaints_list = COMPLAINTS_TO_RUN_FROM_SCRIPT
        complaints_to_process_source = "script variable (COMPLAINTS_TO_RUN_FROM_SCRIPT)"
        print(f"ℹ️ Processing complaints specified in script variable: {', '.join(final_complaints_list)}")
    # If final_complaints_list is still None here, it means we process all.

    if final_complaints_list:
        filtered_differentials = {}
        for complaint_name_to_find in final_complaints_list:
            found_match = False
            for existing_complaint_key in differentials_by_complaint.keys():
                if complaint_name_to_find.lower() in existing_complaint_key.lower():
                    filtered_differentials[existing_complaint_key] = differentials_by_complaint[existing_complaint_key]
                    print(f"  ✓ Matched complaint: '{existing_complaint_key}' for target '{complaint_name_to_find}'")
                    found_match = True
            if not found_match:
                print(f"  ⚠️ No match found in loaded differentials for target complaint: '{complaint_name_to_find}' (from {complaints_to_process_source})")
        
        if not filtered_differentials:
            print(f"❌ No complaints to process based on {complaints_to_process_source}. Exiting or using all if fallback intended.")
            # Decide if you want to exit or fall back to all complaints if the list is empty after filtering
            # For now, let's assume if a list was provided (CLI or script) and it results in no matches, we process nothing from it.
            differentials_by_complaint = {}
        else:
            differentials_by_complaint = filtered_differentials
    else:
        print("ℹ️ Processing all available complaints from 'tables_list' directory.")
            
    # Create output directory if it doesn't exist
    output_dir = "artefacts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Track all generated cases
    all_cases = []

    def log_failed_case(complaint_name, differential_name):
        """Helper function to log a failed differential immediately to its complaint-specific file."""
        try:
            # Construct path to complaint-specific directory
            # Ensure sanitize_filename is available or define/import it if not in scope
            # For this edit, assuming sanitize_filename is accessible as it's used elsewhere
            current_complaint_dir = os.path.join("artefacts", sanitize_filename(complaint_name))
            # Ensure this directory exists (it should have been created in the main loop for the complaint)
            os.makedirs(current_complaint_dir, exist_ok=True)

            complaint_failed_log_path = os.path.join(current_complaint_dir, "failed_differentials.jsonl")

            with jsonlines.open(complaint_failed_log_path, mode='a') as writer: # Append mode
                writer.write(differential_name) # Write just the name of the differential
            print(f"📝 Logged failed differential '{differential_name}' for complaint '{complaint_name}' to {complaint_failed_log_path}")
        except Exception as e:
            print(f"❌ Error logging failed differential for {complaint_name} - {differential_name}: {str(e)}")

    # Maximum number of batch attempts per differential
    MAX_BATCH_ATTEMPTS = 3
    
    # For each presenting complaint
    for complaint, differentials in differentials_by_complaint.items():
        print(f"\n🩺 Processing presenting complaint: {complaint}")
        print(f"📋 Found {len(differentials)} differential diagnoses")
        
        # Create a separate folder for each complaint
        complaint_dir = os.path.join(output_dir, sanitize_filename(complaint))
        os.makedirs(complaint_dir, exist_ok=True)
        
        # Track generated diagnoses to avoid duplicates
        generated_diagnoses = set()
        
        # Determine the number of differentials to process
        if max_cases_per_complaint == 0:  # 0 means process all available differentials
            differentials_to_process = differentials
            print(f"⚙️ Processing all {len(differentials_to_process)} differentials for this complaint.")
        else:
            differentials_to_process = differentials[:max_cases_per_complaint]
            print(f"⚙️ Processing up to {max_cases_per_complaint} differentials (found {len(differentials_to_process)} of {len(differentials)} total for this complaint).")
        
        # Generate a case for each differential diagnosis
        for differential in differentials_to_process:
            # Skip if we've already generated a case for this diagnosis
            if differential in generated_diagnoses:
                print(f"⏩ Skipping {differential} - already generated")
                continue
                
            print(f"\n🔍 Generating case for: {differential}")
            
            # Track attempts for this differential
            attempts = 0
            case_generated = False
            
            while not case_generated and attempts < MAX_BATCH_ATTEMPTS:
                attempts += 1
                print(f"  🌀 Attempt {attempts}/{MAX_BATCH_ATTEMPTS}")
                
                # Create prompt for this differential
                prompt = f"""

                
You are a clinical assistant generating synthetic training data of medical cases. Your output will be parsed as JSON, so **valid JSON formatting is critical**.

📌 TASK: Create a detailed medical case about {differential} for a {complaint} complaint.

Include:
1. **Pertinent positive AND negative findings** in:
   - History (e.g. denies urinary symptoms, no past abdominal surgery)
   - Examination (e.g. no focal neurology, no hepatosplenomegaly)
   - Investigations (e.g. normal ECG, negative troponin, normal lipase)

   EXTREMELY IMPORTANT INSTRUCTIONS:
1. Return ONLY a valid JSON object, nothing else.
2. Do NOT include ```json or ``` tags around your output.
3. Make sure all strings are enclosed in double quotes, not single quotes.
4. Do NOT add any comments or extra text before or after the JSON.
5. Ensure EVERY string is properly closed with a matching quote.
6. Make sure your JSON is valid and can be parsed by standard JSON parsers.
7. Use the EXACT field names shown in the example below.

2. These findings should help **narrow the differential diagnosis** — by ruling out common or dangerous alternatives.

3. All investigations ordered should be **relevant to the presenting complaint** and help confirm the diagnosis of {differential}, or rule out other differentials.

For this case, create a realistic presentation with an adequate amount of medical uncertainty that would lead to a diagnosis of {differential}.

**Refrain from including pathognomonic signs of diseases in the case**

**Refrain from creating history and exam findings that are too obvious**

ESPECIALLY in the history, make it less obvious. Make the symptoms of the correct diagnosis less classical. 

**Reduce the number of typical symptoms and signs**.

Make the diagnostic process difficult.

Difficulty can be increased in cases in the form of **more diagnostically complicated (e.g. red herrings, findings or investigations that don't fit typically into the case)**, **less typical symptoms/signs**, **subtle or absent signs**.

Please generate a detailed and realistic OSCE-style medical case that focuses on {differential} as the correct diagnosis.

Return only valid JSON. No explanation text before or after the JSON object. 

Use the EXACT field names and EXACT ORDER AND EXACT STRUCTURE as shown below. 

Provide the diagnosis only at the end of the case.
Example format:
{example_json_format}

"""

                # This outer try-except block catches errors during the model call itself
                # or fundamental issues with the response object.
                try:
                    print("\n📝 Sending prompt to model...")
                    
                    response = model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 1.0,
                            "max_output_tokens": 8000,
                            "top_k": 40,
                            "top_p": 0.8,
                        }
                    )
                    
                    # Validate the model's response:
                    # Ensure the response object exists, has a 'text' attribute, and the text is not empty.
                    if not response or not hasattr(response, 'text') or not response.text:
                        print("❌ Invalid or empty response from model")
                        # Save the prompt that led to the empty/invalid response for debugging.
                        prompt_dir = os.path.join(output_dir, "debug_prompts")
                        os.makedirs(prompt_dir, exist_ok=True)
                        prompt_filename = f"{sanitize_filename(complaint)}__{sanitize_filename(differential)}_attempt{attempts}.txt"
                        prompt_filepath = os.path.join(prompt_dir, prompt_filename)
                        
                        with open(prompt_filepath, 'w', encoding='utf-8') as f:
                            f.write(prompt)
                        
                        print(f"📝 Saved problematic prompt to {prompt_filepath}")
                        continue # Move to the next attempt for this differential.
                    
                    response_text = response.text
                    print(f"📄 Raw response length: {len(response_text)} characters")
                    # Print first 100 characters of response for debugging, removing newlines for cleaner log output.
                    print(f"📄 Response begins with: {response_text[:100].replace(chr(10), '').replace(chr(13), '')}...")
                    
                    # Save the complete raw response from the model to a file.
                    # This is useful for inspecting exactly what the model returned before any processing.
                    raw_dir = os.path.join(output_dir, "raw_responses")
                    os.makedirs(raw_dir, exist_ok=True)
                    raw_filename = f"{sanitize_filename(complaint)}__{sanitize_filename(differential)}_attempt{attempts}.txt"
                    raw_filepath = os.path.join(raw_dir, raw_filename)
                    
                    with open(raw_filepath, 'w', encoding='utf-8') as f:
                        f.write(response_text)
                    
                    print(f"📝 Saved raw response to {raw_filepath}")
                    
                    # --- Start of JSON Cleaning Steps ---
                    # These steps attempt to pre-process the raw text to make it more likely to be valid JSON.
                    cleaned_text = response_text
                    print("🧹 Cleaning response...")
                    
                    # Remove markdown code block fences (e.g., ```json ... ``` or ``` ... ```).
                    # The model sometimes wraps its JSON output in these fences.
                    # Using re.DOTALL allows .*? to match across multiple lines.
                    markdown_match = re.search(r"^\\s*```(?:json)?\\s*\\n?(.*?)\\n?\\s*```\\s*$", response_text, re.DOTALL)
                    if markdown_match:
                        cleaned_text = markdown_match.group(1) # Extract content within the fences.
                        print("  ✓ Removed markdown code block using regex")
                    else:
                        # Fallback: If regex doesn't match, try simple prefix/suffix removal.
                        # This handles cases where the model might output, e.g., ```json ... (and forgets closing ```).
                        if response_text.startswith("```json"):
                            cleaned_text = response_text[len("```json"):]
                            if cleaned_text.startswith("\\n"): # Remove potential leading newline
                                cleaned_text = cleaned_text[1:]
                            print("  ✓ Removed ```json prefix (fallback)")
                        elif response_text.startswith("```"):
                            cleaned_text = response_text[len("```"):]
                            if cleaned_text.startswith("\\n"): # Remove potential leading newline
                                cleaned_text = cleaned_text[1:]
                            print("  ✓ Removed ``` prefix (fallback)")

                        # Always try to remove a trailing ``` if it exists and wasn't caught by the regex.
                        if cleaned_text.endswith("```"):
                            cleaned_text = cleaned_text[:-len("```")]
                            if cleaned_text.endswith("\\n"): # Remove potential trailing newline
                                cleaned_text = cleaned_text[:-1]
                            print("  ✓ Removed ``` suffix (fallback)")
                    
                    # Remove any leading or trailing whitespace that might remain or was outside fences.
                    original_length = len(cleaned_text)
                    cleaned_text = cleaned_text.strip()
                    if len(cleaned_text) != original_length:
                        print(f"  ✓ Stripped leading/trailing whitespace ({original_length - len(cleaned_text)} chars)")

                    # Attempt to fix common truncation issues before formal parsing.
                    # Check for imbalanced curly braces (often indicates a truncated JSON object).
                    open_braces = cleaned_text.count('{')
                    close_braces = cleaned_text.count('}')
                    
                    if open_braces != close_braces:
                        print(f"⚠️ Detected potential truncation: {open_braces} opening braces, {close_braces} closing braces")
                        if open_braces > close_braces:
                            # If truncated, append the missing closing braces.
                            cleaned_text += "}" * (open_braces - close_braces)
                            print(f"🔧 Added {open_braces - close_braces} closing braces to fix truncation")
                    
                    # Check for a string value that appears truncated at the very end of the JSON.
                    # Example: "key": "value without closing quote
                    string_pattern = r'"([^"]+)"\\s*:\\s*"([^"]+)$'
                    if re.search(string_pattern, cleaned_text):
                        print("⚠️ Detected truncated string value at the end of JSON")
                        # Append a closing double quote to fix the truncated string.
                        cleaned_text += '"'
                        print("🔧 Added closing quote to fix truncated string")
                    # --- End of initial pre-parsing JSON Cleaning Steps ---

                    # Attempt to parse the cleaned text as JSON.
                    # This is the first attempt using the standard json.loads().
                    try:
                        parsed_full_case = json.loads(cleaned_text)
                        
                        # If the model returns a list (e.g., `[{...}]`) instead of a single object,
                        # and it contains exactly one element, extract that element.
                        if isinstance(parsed_full_case, list) and len(parsed_full_case) == 1:
                            print("ℹ️ Model returned an array, extracting the first object.")
                            parsed_full_case = parsed_full_case[0]
                        elif isinstance(parsed_full_case, list):
                            # If it's a list with multiple elements, it's not the expected format.
                            print(f"❌ Expected a JSON object, but got an array with {len(parsed_full_case)} elements after initial parse.")
                            continue # To next attempt for this differential.
                            
                        # Ensure the parsed result is a dictionary (JSON object) as expected.
                        if not isinstance(parsed_full_case, dict):
                            print(f"❌ Expected a JSON object after initial parse, but got type {type(parsed_full_case)}.")
                            continue # To next attempt.
                        
                        # Validate presence and type of critical top-level keys.
                        original_complaint_value = parsed_full_case.get("Presenting complaint")
                        if not original_complaint_value or not isinstance(original_complaint_value, str):
                            print("❌ Missing, empty, or invalid type for 'Presenting complaint' in initial parse.")
                            continue # To next attempt.
                            
                        actual_case_content = parsed_full_case.get("case")
                        if not isinstance(actual_case_content, dict):
                            print("❌ Missing, empty, or incorrect type for field: 'case' (must be an object) in initial parse.")
                            continue # To next attempt.
                        
                        # Perform specific key nesting validation on the extracted 'case' content.
                        # This function (validate_specific_key_nesting) ensures that the critical
                        # diagnostic path (OSCE_Examination -> ... -> Correct_Diagnosis) exists.
                        validation_passed, validation_msg = validate_specific_key_nesting(actual_case_content)
                        
                        if not validation_passed:
                            # If the required nested structure is not found, log the failure and try next attempt.
                            print(f"❌ Specific key nesting validation failed for {complaint} - {differential}: {validation_msg}")
                            log_failed_case(complaint, differential)
                            continue # To next attempt.
                        else:
                            print(f"✅ {validation_msg}")

                        # --- JSON Parsing and Validation Successful (First Attempt) ---
                        # If all checks pass, the case is considered successfully generated.
                        
                        generated_diagnoses.add(differential) # Track to avoid duplicates for this complaint.
                        
                        # Get the diagnosis category (tag)
                        diagnosis_tag = diagnosis_categories_by_complaint.get(complaint, {}).get(differential, "Unknown")
                        
                        # Add tag field to the case content
                        tagged_case_content = copy.deepcopy(actual_case_content)
                        tagged_case_content["tag"] = diagnosis_tag
                        
                        all_cases.append({ # Add to the list of all successfully generated cases.
                            "intended_complaint_category": complaint, # Store the category it was generated for
                            "ai_presenting_complaint": original_complaint_value, # Keep AI's version
                            "content_to_write": tagged_case_content, # Store the case object for output with tag
                            "diagnosis_category": diagnosis_tag
                        })
                        
                        # Save the successfully parsed and validated 'case' object to its own JSON file.
                        case_filename = f"{sanitize_filename(differential)}.json"
                        case_filepath = os.path.join(complaint_dir, case_filename)
                        
                        with open(case_filepath, 'w', encoding='utf-8') as f:
                            json.dump(tagged_case_content, f, indent=2, ensure_ascii=False)
                            
                        print(f"✅ Generated and saved case for {differential} (extracted content only)")
                        case_generated = True # Flag that this differential is done, exit the while loop.
                        
                    except json.JSONDecodeError as e: # This block executes if json.loads(cleaned_text) fails.
                        print(f"❌ JSON parse error: {str(e)}")
                        # Show the problematic character and its context if position is available.
                        if hasattr(e, 'pos'):
                            error_context_start = max(0, e.pos - 30)
                            error_context_end = min(len(cleaned_text), e.pos + 30)
                            error_context = cleaned_text[error_context_start:error_context_end]
                            print(f"  Problem near: '...{error_context}...'")
                            print(f"  Error position: {e.pos}")
                        
                        print("🔧 Attempting to fix JSON using json-repair library...")
                        
                        # Second attempt at parsing, this time using the `json_repair` library,
                        # which is more tolerant of common JSON errors.
                        try:
                            # `json_repair` attempts to fix and parse the string directly into Python objects.
                            repaired_full_case = json_repair(cleaned_text, return_objects=True)
                            print("✅ Fixed JSON parsed successfully by json-repair!")
                            
                            # Similar to the first parsing attempt, handle if the repaired output is an array.
                            if isinstance(repaired_full_case, list) and len(repaired_full_case) == 1:
                                print("ℹ️ Repaired JSON was an array, extracting the first object.")
                                repaired_full_case = repaired_full_case[0]
                            elif isinstance(repaired_full_case, list):
                                print(f"❌ Repaired JSON was an array with {len(repaired_full_case)} elements. Skipping this attempt.")
                                continue # To next attempt for this differential.
                                
                            # Ensure the repaired result is a dictionary.
                            if not isinstance(repaired_full_case, dict):
                                print(f"❌ Repaired JSON is not an object, but type {type(repaired_full_case)}. Skipping this attempt.")
                                continue # To next attempt.

                            # Validate critical top-level keys in the repaired JSON.
                            original_complaint_value_repaired = repaired_full_case.get("Presenting complaint")
                            if not original_complaint_value_repaired or not isinstance(original_complaint_value_repaired, str):
                                print(f"❌ Repaired JSON is missing, empty, or invalid type for 'Presenting complaint'. Skipping this attempt.")
                                continue # To next attempt.
                                
                            actual_case_content_repaired = repaired_full_case.get("case")
                            if not isinstance(actual_case_content_repaired, dict):
                                print(f"❌ Repaired JSON is missing, empty, or incorrect type for field: 'case' (must be an object). Skipping this attempt.")
                                continue # To next attempt.

                            # Perform specific key nesting validation on the 'case' content from the repaired JSON.
                            validation_passed_repaired, validation_msg_repaired = validate_specific_key_nesting(actual_case_content_repaired)

                            if not validation_passed_repaired:
                                # If validation fails even after repair, log and try next attempt.
                                print(f"❌ Specific key nesting validation failed for REPAIRED {complaint} - {differential}: {validation_msg_repaired}")
                                log_failed_case(complaint, differential)
                                continue # To next attempt.
                            else:
                                print(f"✅ {validation_msg_repaired} (repaired JSON)")

                            # --- JSON Repair and Validation Successful ---

                            # For debugging, save the full JSON object *as returned by json_repair* before extracting the 'case'.
                            fixed_dir = os.path.join(output_dir, "fixed_json_originals") 
                            os.makedirs(fixed_dir, exist_ok=True)
                            fixed_filename = f"{sanitize_filename(complaint)}__{sanitize_filename(differential)}_attempt{attempts}_repaired_original.json"
                            fixed_filepath = os.path.join(fixed_dir, fixed_filename)
                            
                            with open(fixed_filepath, 'w', encoding='utf-8') as f:
                                json.dump(repaired_full_case, f, indent=2, ensure_ascii=False)
                                
                            print(f"📝 Saved *original* repaired JSON (before extraction) to {fixed_filepath}")
                            
                            # Get the diagnosis category (tag)
                            diagnosis_tag = diagnosis_categories_by_complaint.get(complaint, {}).get(differential, "Unknown")
                            
                            # Add tag field to the case content
                            tagged_case_content_repaired = copy.deepcopy(actual_case_content_repaired)
                            tagged_case_content_repaired["tag"] = diagnosis_tag
                            
                            # Add to generated diagnoses and all_cases list.
                            generated_diagnoses.add(differential)
                            all_cases.append({
                                "intended_complaint_category": complaint, # Store the category it was generated for
                                "ai_presenting_complaint": original_complaint_value_repaired, # Keep AI's version
                                "content_to_write": tagged_case_content_repaired,
                                "diagnosis_category": diagnosis_tag
                            })
                            
                            # Save the successfully repaired and validated 'case' object to its JSON file.
                            case_filename = f"{sanitize_filename(differential)}.json"
                            case_filepath = os.path.join(complaint_dir, case_filename)
                            with open(case_filepath, 'w', encoding='utf-8') as f:
                                json.dump(tagged_case_content_repaired, f, indent=2, ensure_ascii=False)
                                
                            print(f"✅ Generated and saved case for {differential} from repaired JSON (extracted content only)")
                            case_generated = True # Flag successful generation.
                            # No 'continue' here; processing for this differential is complete.

                        except Exception as e2: # This catches errors from json_repair or subsequent logic.
                            print(f"❌ Failed to fix JSON with json-repair or process it: {str(e2)}")
                            # If json-repair itself or subsequent validation fails, this attempt is considered failed.
                            # The loop will continue to the next attempt for this differential.
                
                except Exception as e: # Outer except: Catches errors from model.generate_content() or initial response handling.
                    print(f"❌ Error generating content: {str(e)}")
                    print(f"  Exception type: {type(e).__name__}")
                    
                    # Add a small delay before retrying to respect potential rate limits or transient issues.
                    time.sleep(2)
            
            # This code executes after all attempts for a single differential diagnosis are exhausted
            # or if a case was successfully generated (case_generated = True).
            if not case_generated:
                # If, after all MAX_BATCH_ATTEMPTS, no valid case was generated for this differential.
                print(f"⚠️ Failed to generate case for {differential} after {MAX_BATCH_ATTEMPTS} attempts")
                log_failed_case(complaint, differential)
        
        # After processing all differentials for the current presenting complaint:
        # Save all successfully generated and extracted 'case' objects for this complaint to a single JSONL file.
        # JSONL (JSON Lines) format is one JSON object per line.
        complaint_cases_to_write = [
            item["content_to_write"] for item in all_cases
            # Filter all_cases to include only those generated for the current complaint category.
            if item["intended_complaint_category"].lower() == complaint.lower()
        ]
        jsonl_filename = f"{sanitize_filename(complaint)}_all_cases.jsonl"
        jsonl_filepath = os.path.join(output_dir, jsonl_filename) # Saved in the main output_dir, not complaint specific subdir.
        
        if complaint_cases_to_write: # Only write the file if there are cases for this complaint.
            with jsonlines.open(jsonl_filepath, mode='w') as writer:
                for case_content_item in complaint_cases_to_write:
                    writer.write(case_content_item)
            print(f"✅ Saved {len(complaint_cases_to_write)} extracted cases for {complaint} to {jsonl_filepath}")
        else:
            print(f"ℹ️ No cases to write for complaint '{complaint}' to {jsonl_filepath} (either no cases generated or filtering mismatch).")

    # After processing all complaints:
    # Save all generated cases from *all* complaints to a single global JSONL file.
    all_cases_filepath = os.path.join(output_dir, "all_cases.jsonl")
    if all_cases: # Only write if there are any cases at all
        with jsonlines.open(all_cases_filepath, mode='w') as writer:
            for item in all_cases:
                writer.write(item["content_to_write"])
        print(f"✅ Saved all {len(all_cases)} extracted cases to global {all_cases_filepath}")
    else:
        print(f"ℹ️ No cases generated overall to write to global {all_cases_filepath}.")
            
    print(f"\n✅ Done! Generated {len(all_cases)} cases across {len(differentials_by_complaint)} presenting complaints.")
    return all_cases

# Example usage
if __name__ == "__main__":
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(description="Generate medical cases from differential diagnoses")
    parser.add_argument("--max_cases", type=int, default=0, 
                        help="Maximum number of cases to generate per complaint. Specify 0 or omit to process all differentials for each complaint (default: 0).")
    parser.add_argument("--complaints", type=str, nargs='+',
                        help="Specific complaints to process (if not specified, all complaints will be processed)")
    
    args = parser.parse_args()
    
    if args.max_cases == 0:
        print("🔍 Running with max_cases_per_complaint = 0 (process all differentials for each complaint).")
    else:
        print(f"🔍 Running with max_cases_per_complaint = {args.max_cases}")
    
    # No need to print about args.complaints here, it's handled inside generate_cases_from_differentials
    # if args.complaints:
    #     print(f"🔍 Processing only these complaints: {', '.join(args.complaints)}")
    
    all_generated_cases = generate_cases_from_differentials(
        max_cases_per_complaint=args.max_cases,
        specific_complaints=args.complaints
    )

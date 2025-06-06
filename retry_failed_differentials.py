import os
import json
import jsonlines
import re
import time
import copy
from vertexai.preview.generative_models import GenerativeModel
import vertexai
from dotenv import load_dotenv
from json_repair import repair_json as json_repair

#python retry_failed_differentials.py
# --- Helper functions (copied from main_generation_backup.py) ---
def sanitize_filename(name):
    invalid_chars = r'[\\/:*?"<>|]'
    return re.sub(invalid_chars, '_', name)

def normalize_differential_name(name):
    # Lowercase, strip prefixes, replace spaces/underscores with single underscore, remove non-alphanumeric except underscores
    name = strip_prefix(name)
    name = name.lower()
    name = re.sub(r'[\s_]+', '_', name)  # Replace spaces and underscores with single underscore
    name = re.sub(r'[^a-z0-9_]', '', name)  # Remove all non-alphanumeric except underscore
    name = re.sub(r'_+', '_', name)  # Collapse multiple underscores
    return name.strip('_')

def strip_prefix(differential):
    prefixes = [
        'vascular:', 'infection:', 'cancer:', 'other:', 'rarity:', 'pulmonary cause:',
        'neoplasia/cancer:', 'pitfalls:', 'serious disorders:', 'probability diagnosis:',
        'masquerades:', 'patient trying to tell me something:', 'pitfall:', 'serious disorder:'
    ]
    d = differential.lower()
    for prefix in prefixes:
        if d.startswith(prefix):
            return differential[len(prefix):].strip()
    return differential

def strip_suffix(name):
    # Remove (n) suffixes before .json
    return re.sub(r'\s*\(\d+\)$', '', name)

def validate_specific_key_nesting(data_dict, base_path_name="case"):
    osce_key = "OSCE_Examination"
    osce_path_str = f"{base_path_name}.{osce_key}"
    if not isinstance(data_dict, dict):
        return False, f"Validation Error: Expected '{base_path_name}' to be a dictionary, but found {type(data_dict).__name__}."
    if osce_key not in data_dict:
        return False, f"Validation Error: Missing key '{osce_key}' at path '{base_path_name}' (expected '{osce_path_str}')."
    osce_level_dict = data_dict[osce_key]
    if not isinstance(osce_level_dict, dict):
        return False, f"Validation Error: Key '{osce_key}' at path '{osce_path_str}' should point to a dictionary, but found {type(osce_level_dict).__name__}."
    expected_children_of_osce = [
        ("Patient_Actor", True),
        ("Physical_Examination_Findings", True),
        ("Test_Results", True),
        ("Correct_Diagnosis", False)
    ]
    children_base_path_str = osce_path_str
    for child_key, must_be_dict in expected_children_of_osce:
        current_child_path_str = f"{children_base_path_str}.{child_key}"
        if child_key not in osce_level_dict:
            return False, f"Validation Error: Missing key '{child_key}' at path '{current_child_path_str}'."
        if must_be_dict:
            child_value = osce_level_dict[child_key]
            if not isinstance(child_value, dict):
                return False, f"Validation Error: Key '{child_key}' at path '{current_child_path_str}' should point to a dictionary, but found {type(child_value).__name__}."
    return True, f"Specific key nesting validated successfully ({osce_path_str} and its direct children)."

def all_normalized_forms(name):
    forms = set()
    forms.add(normalize_differential_name(name))
    forms.add(normalize_differential_name(strip_prefix(name)))
    return forms

# Example JSON format (truncated for brevity, use your full example in production)
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

# --- Model Initialization ---
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"
vertexai.init(location="us-central1", credentials=None)
model = GenerativeModel("gemini-2.5-pro-preview-03-25")

# --- Main retry logic ---
ARTEFACTS_DIR = "artefacts"
MAX_BATCH_ATTEMPTS = 15

def retry_failed_differentials():
    # Load all diagnoses first
    diagnoses_path = os.path.join("tables_list", "diagnoses.jsonl")
    all_diagnoses = {}
    if os.path.exists(diagnoses_path):
        with jsonlines.open(diagnoses_path, 'r') as reader:
            for entry in reader:
                for complaint_key in entry:
                    normalized_key = normalize_differential_name(complaint_key)
                    all_diagnoses[normalized_key] = entry[complaint_key]

    for complaint_folder in os.listdir(ARTEFACTS_DIR):
        complaint_path = os.path.join(ARTEFACTS_DIR, complaint_folder)
        if not os.path.isdir(complaint_path):
            continue
            
        # Skip hidden directories and system folders
        if complaint_folder.startswith('.') or complaint_folder == '__pycache__':
            continue
            
        # Check for failed differentials
        failed_path = os.path.join(complaint_path, "failed_differentials.jsonl")
        
        # Get the normalized complaint name for lookup
        normalized_complaint = normalize_differential_name(complaint_folder)
        
        # Skip if no failed differentials and no diagnoses found
        if not os.path.exists(failed_path) and normalized_complaint not in all_diagnoses:
            print(f"  ⚠️ Skipping {complaint_folder}: no failed_differentials.jsonl and no matching diagnoses entry")
            continue
            
        print(f"\n🔄 Retrying failed differentials for: {complaint_folder}")
        # Extract complaint from folder name
        complaint = complaint_folder.replace('_', ' ').title()
        
        # Always build category_map from diagnoses list for this complaint
        category_map = {}
        diag_entry = all_diagnoses.get(normalized_complaint, {})
        if isinstance(diag_entry, dict):
            for k, v in diag_entry.items():
                if k not in ["Masquerades", "Patient trying to tell me something"] and isinstance(v, list):
                    for diff in v:
                        norm_diff = normalize_differential_name(diff)
                        if norm_diff not in category_map:
                            category_map[norm_diff] = k
        if os.path.exists(failed_path):
            # Read failed differentials
            with open(failed_path, 'r', encoding='utf-8') as f:
                failed_diffs = [line.strip().strip('"') for line in f if line.strip()]
        else:
            # Use diagnoses from the all_diagnoses dictionary
            print("  (No failed_differentials.jsonl found, using diagnoses list)")
            if isinstance(diag_entry, dict):
                failed_diffs = []
                for k, v in diag_entry.items():
                    if k not in ["Masquerades", "Patient trying to tell me something"] and isinstance(v, list):
                        for diff in v:
                            failed_diffs.append(diff)
            else:
                failed_diffs = diag_entry if isinstance(diag_entry, list) else []
            
        if not failed_diffs:
            print("  (No differentials found)")
            continue
        
        # Remove duplicates while preserving order (using enhanced normalization)
        seen = set()
        filtered_diffs = []
        for x in failed_diffs:
            norm_x = normalize_differential_name(x)
            if norm_x not in seen:
                seen.add(norm_x)
                filtered_diffs.append(x)
        failed_diffs = filtered_diffs
        
        # Normalize all existing filenames (without .json extension, using enhanced normalization, and strip suffixes)
        existing_files_normalized = set()
        for f in os.listdir(complaint_path):
            if f.lower().endswith('.json'):
                base = strip_suffix(f[:-5])
                existing_files_normalized.update(all_normalized_forms(base))
        filtered_diffs = []
        for diff in failed_diffs:
            norm_forms = all_normalized_forms(diff)
            if any(form in existing_files_normalized for form in norm_forms):
                print(f"  ⏩ Skipping {diff} (already present as a .json file)")
            else:
                filtered_diffs.append(diff)
        failed_diffs = filtered_diffs
        
        if not failed_diffs:
            print("  (No new failed differentials to process)")
            continue
        
        print(f"  Processing {len(failed_diffs)} unique failed differentials")
        
        # Prepare all_cases.jsonl path
        all_cases_path = os.path.join(ARTEFACTS_DIR, f"{complaint_folder}_all_cases.jsonl")
        # Load existing cases to avoid duplicates
        existing_cases = []
        if os.path.exists(all_cases_path):
            with jsonlines.open(all_cases_path, 'r') as reader:
                existing_cases = [case for case in reader]
        existing_diagnoses = set()
        for case in existing_cases:
            diagnosis = case.get("Correct_Diagnosis") or (case.get("case", {}).get("OSCE_Examination", {}).get("Correct_Diagnosis"))
            if diagnosis:
                existing_diagnoses.add(diagnosis)
        # Retry each failed differential
        still_failed = []
        for differential in failed_diffs:
            if differential in existing_diagnoses:
                print(f"  ⏩ Skipping {differential} (already present)")
                continue
            print(f"  🔍 Generating case for: {differential}")
            attempts = 0
            case_generated = False
            while not case_generated and attempts < MAX_BATCH_ATTEMPTS:
                attempts += 1
                print(f"    🌀 Attempt {attempts}/{MAX_BATCH_ATTEMPTS}")
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


                try:
                    response = model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 1.0,
                            "max_output_tokens": 8000,
                            "top_k": 40,
                            "top_p": 0.8,
                        }
                    )
                    response_text = response.text if hasattr(response, 'text') else ''
                    cleaned_text = response_text.strip()
                    # Remove markdown code block if present
                    if cleaned_text.startswith('```json'):
                        cleaned_text = cleaned_text[len('```json'):].strip()
                    elif cleaned_text.startswith('```'):
                        cleaned_text = cleaned_text[len('```'):].strip()
                    if cleaned_text.endswith('```'):
                        cleaned_text = cleaned_text[:-3].strip()
                    # Fix braces
                    open_braces = cleaned_text.count('{')
                    close_braces = cleaned_text.count('}')
                    if open_braces > close_braces:
                        cleaned_text += '}' * (open_braces - close_braces)
                    # Try parsing
                    try:
                        parsed_case = json.loads(cleaned_text)
                    except Exception:
                        parsed_case = json_repair(cleaned_text, return_objects=True)
                    # If list, extract first
                    if isinstance(parsed_case, list) and len(parsed_case) == 1:
                        parsed_case = parsed_case[0]
                    # Validate
                    presenting_complaint = parsed_case.get("Presenting complaint")
                    case_content = parsed_case.get("case")
                    if not presenting_complaint or not isinstance(presenting_complaint, str):
                        print("      ❌ Missing or invalid 'Presenting complaint'")
                        continue
                    if not isinstance(case_content, dict):
                        print("      ❌ Missing or invalid 'case' object")
                        continue
                    valid, msg = validate_specific_key_nesting(case_content)
                    if not valid:
                        print(f"      ❌ Structure validation failed: {msg}")
                        continue
                    # Save as .json
                    # Add tag to case content
                    norm_diff = normalize_differential_name(differential)
                    tag = category_map.get(norm_diff, "Unknown")
                    case_content_with_tag = copy.deepcopy(case_content)
                    case_content_with_tag["tag"] = tag
                    # Save tagged case
                    case_filename = f"{normalize_differential_name(differential)}.json"
                    case_filepath = os.path.join(complaint_path, case_filename)
                    with open(case_filepath, 'w', encoding='utf-8') as f:
                        json.dump(case_content_with_tag, f, indent=2, ensure_ascii=False)
                    # Append to all_cases.jsonl
                    with jsonlines.open(all_cases_path, mode='a') as writer:
                        writer.write(case_content_with_tag)
                    print(f"      ✅ Generated and saved case for {differential}")
                    case_generated = True
                except Exception as e:
                    print(f"      ❌ Error: {str(e)}")
                    time.sleep(2)
            if not case_generated:
                still_failed.append(differential)
        # Update failed_differentials.jsonl
        with open(failed_path, 'w', encoding='utf-8') as f:
            for diff in still_failed:
                f.write(f'"{diff}"\n')
        print(f"  Remaining failed: {len(still_failed)}")

if __name__ == "__main__":
    retry_failed_differentials() 
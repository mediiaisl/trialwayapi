import requests
import torch
import re
import spacy
from scipy.spatial.distance import euclidean
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from flask import Flask, request, jsonify

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

def expand_contractions(text):
    contractions = {"don't": "do not", "won't": "will not", "can't": "cannot"}
    contractions_pattern = re.compile('({})'.format('|'.join(contractions.keys())), flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contractions.get(match.lower() if match.lower() in contractions else match.lower().split("'")[0])
        return first_char + expanded_contraction[1:]
    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text

# Function to preprocess text
def preprocess_text(text, context_window_size=None):
    """
    Preprocesses the text by expanding contractions, converting to lower case,
    removing special characters, and optionally adjusting the context window size.
    """
    text = expand_contractions(text.lower())
    text = re.sub(r'[^a-z0-9.,; ]', '', text)
    doc = nlp(text)

    # Implement logic for context window size
    if context_window_size is not None:
        words = [token.text for token in doc]
        if len(words) > context_window_size:
            words = words[:context_window_size]
        doc = nlp(' '.join(words))

    return " ".join([token.lemma_ for token in doc])

def fetch_studies(expr, fields, max_per_request, cities):
    base_url = "https://clinicaltrials.gov/api/query/study_fields"
    fmt = "json"
    current_rank = 1
    nct_ids = []

    while current_rank is not None:
        params = {"expr": expr, "fields": ','.join(fields), "min_rnk": current_rank, "max_rnk": current_rank + max_per_request - 1, "fmt": fmt}
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            break

        data = response.json().get("StudyFieldsResponse")
        if not data:
            break

        current_rank += data["NStudiesReturned"]

        for study in data["StudyFields"]:
            nct_id = study.get("NCTId", [None])[0]
            if not nct_id or study.get("StudyType", [None])[0] == "Observational":
                continue

            if study.get("OverallStatus", [None])[0] not in ["Recruiting", "Not yet recruiting"]:
                continue

            if any(city in study.get("LocationCity", []) for city in cities):
                nct_ids.append(nct_id)

        if current_rank >= data["NStudiesFound"]:
            current_rank = None

    return nct_ids

def fetch_study_details(nct_id):
    details_url = "https://clinicaltrials.gov/api/query/full_studies"
    params = {"expr": nct_id, "fmt": "json"}
    response = requests.get(details_url, params=params)
    if response.status_code != 200:
        return None

    try:
        study_details = response.json()['FullStudiesResponse']['FullStudies'][0]['Study']
        return study_details.get('ProtocolSection', {}).get('EligibilityModule', {}).get('EligibilityCriteria', None)
    except KeyError:
        return None


    



def split_eligibility_criteria(criteria):
    parts = re.split(r'\n\n(Exclusion Criteria:)\n\n', criteria)
    inclusion_criteria = parts[0].replace("Inclusion Criteria:\n\n", "").strip()
    exclusion_criteria = parts[-1].replace("Exclusion Criteria:\n\n", "").strip() if len(parts) > 1 else ""
    return inclusion_criteria, exclusion_criteria



def get_embeddings(text, context_window_size=None):
    """
    Generate embeddings using the SentenceTransformer model with preprocessing.
    """
    processed_text = preprocess_text(text, context_window_size)
    return model.encode(processed_text)

    

def analyze_exclusion_criteria(special_conditions, exclusion_criteria):
    """
    Analyze the exclusion criteria using SentenceTransformer embeddings and euclidean similarity.
    """
    doc = nlp(exclusion_criteria)
    phrases = [chunk.text for chunk in doc.noun_chunks]
    phrase_embeddings = [get_embeddings(phrase) for phrase in phrases]

    overall_compliance = 'Compliant'
    total_similarity_score = 0
    num_conditions = len(special_conditions)
    details = []

    threshold = 0.40  # Selected similarity threshold

    for condition in special_conditions:
        condition_embedding = get_embeddings(condition)
        highest_similarity = 0
        most_similar_phrase = "None"

        for phrase, embedding in zip(phrases, phrase_embeddings):
            similarity = 1 - euclidean(condition_embedding, embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_phrase = phrase

        compliance = 'Not Compliant' if highest_similarity > threshold else 'Compliant'
        total_similarity_score += highest_similarity

        if compliance == 'Not Compliant':
            overall_compliance = 'Not Compliant'

        details.append({
            'condition': condition,
            'compliance': compliance,
            'trust_score': highest_similarity * 100,
            'most_similar_phrase': most_similar_phrase,
            'similarity': highest_similarity
        })

    overall_compliance_score = (total_similarity_score / num_conditions) * 100

    explanation = f"Overall Compliance: {overall_compliance}. "
    for detail in details:
        explanation += (f"Condition '{detail['condition']}' is {detail['compliance']} "
                        f"(most similar phrase: '{detail['most_similar_phrase']}', "
                        f"similarity: {detail['similarity']:.2f}, "
                        f"trust score: {detail['trust_score']:.2f}). ")

    return overall_compliance, overall_compliance_score, explanation

def split_eligibility_criteria(criteria):
    parts = re.split(r'\n\n(Exclusion Criteria:)\n\n', criteria)
    inclusion_criteria = parts[0].replace("Inclusion Criteria:\n\n", "").strip()
    exclusion_criteria = parts[-1].replace("Exclusion Criteria:\n\n", "").strip() if len(parts) > 1 else ""
    return inclusion_criteria, exclusion_criteria

def analyze_inclusion_criteria(special_conditions, inclusion_criteria, threshold=0.5):
    """
    Analyze the inclusion criteria using SentenceTransformer and advanced NLP techniques.
    """
    sentences = [sent.text.strip() for sent in nlp(inclusion_criteria).sents]
    sentence_embeddings = [get_embeddings(sentence) for sentence in sentences]

    # Dynamically assigning equal weights to each special condition
    condition_weights = {condition: 1.0 for condition in special_conditions}

    total_weight = sum(condition_weights.values())
    weighted_inclusion_probability = 0.0
    weighted_inclusion_trust_score = 0.0

    for condition in special_conditions:
        condition_embedding = get_embeddings(condition)
        condition_weight = condition_weights[condition]

        condition_probability = 0.0
        condition_max_similarity = 0.0

        for sentence, sentence_embedding in zip(sentences, sentence_embeddings):
            similarity = cosine_similarity([condition_embedding], [sentence_embedding])[0][0]
            if similarity > condition_max_similarity:
                condition_max_similarity = similarity
                condition_probability = similarity if similarity >= threshold else 0

        weighted_inclusion_probability += condition_probability * condition_weight
        weighted_inclusion_trust_score += condition_max_similarity * 100 * condition_weight

    overall_inclusion_probability = weighted_inclusion_probability / total_weight
    overall_inclusion_trust_score = weighted_inclusion_trust_score / total_weight

    return overall_inclusion_probability, overall_inclusion_trust_score

def fetch_detailed_study_info(nct_id):
    """
    Fetch detailed information for a given study using its NCT ID.
    """
    details_url = "https://clinicaltrials.gov/api/query/full_studies"
    params = {"expr": nct_id, "fmt": "json"}
    response = requests.get(details_url, params=params)

    if response.status_code != 200:
        return None

    try:
        study_details = response.json()['FullStudiesResponse']['FullStudies'][0]['Study']
    
        design_primary_purpose = study_details.get("ProtocolSection", {}).get("DesignModule", {}).get("DesignPrimaryPurpose", "N/A")
        return study_details, design_primary_purpose
    except KeyError:
        return None, "N/A"
    
app = Flask(__name__)
@app.route('/analyze', methods=['GET'])
def analyze():
    disease = request.args.get('disease')
    special_conditions = request.args.getlist('special_conditions')
    cities = request.args.getlist('cities')
    countries = request.args.getlist('countries')

    final_response = main(disease, special_conditions, cities, countries)

    # Return the final_response as JSON
    return jsonify(final_response)
    
def response(study_details):

    study_details = sorted(study_details, key=lambda x: x['DesignPrimaryPurpose'] != "Treatment")
   
    # Sort studies by phase
    phase_order = {"Phase IV": 4, "Phase III": 3, "Phase II": 2, "Phase I": 1, "N/A": 0}
    sorted_studies = sorted(study_details, key=lambda x: phase_order.get(x['Phase'], 0), reverse=True)

    return json.dumps(sorted_studies, indent=4)





def main(disease, special_conditions, cities, countries):
    expr = f"{disease}"
    fields = ["NCTId", "StudyType", "OverallStatus", "LocationCity"]
    max_per_request = 1000
    final_response = []

    nct_ids = fetch_studies(expr, fields, max_per_request, cities)

    study_details_with_purpose = []  # List to store study details along with their DesignPrimaryPurpose

    for nct_id in nct_ids:
        eligibility_criteria = fetch_study_details(nct_id)
        if eligibility_criteria:
            inclusion_criteria, exclusion_criteria = split_eligibility_criteria(eligibility_criteria)

            # Analyze criteria
            compliance, compliance_trust_score, _ = analyze_exclusion_criteria(special_conditions, exclusion_criteria)
            inclusion_probability, inclusion_trust_score = analyze_inclusion_criteria(special_conditions, inclusion_criteria)

            if compliance == 'Compliant':
                # Fetch detailed information along with DesignPrimaryPurpose
                detailed_info, design_primary_purpose = fetch_detailed_study_info(nct_id)
                if detailed_info:
                    protocol_section = detailed_info.get("ProtocolSection", {})
                    identification_module = protocol_section.get("IdentificationModule", {})
                    status_module = protocol_section.get("StatusModule", {})
                    sponsor_module = protocol_section.get("SponsorCollaboratorsModule", {})
                    description_module = protocol_section.get("DescriptionModule", {})
                    design_module = protocol_section.get("DesignModule", {})
                    contacts_module = protocol_section.get("ContactsLocationsModule", {})

                    # Extracting individual fields
                    title = identification_module.get("OfficialTitle", "N/A")
                    status = status_module.get("OverallStatus", "N/A")
                    start_date = status_module.get("StartDateStruct", {}).get("StartDate", "N/A")
                    sponsor = sponsor_module.get("LeadSponsor", {}).get("LeadSponsorName", "N/A")
                    collaborator = sponsor_module.get("ResponsibleParty", {}).get("ResponsiblePartyType", "N/A")
                    description = description_module.get("DetailedDescription", "N/A")
                    phase = design_module.get("PhaseList", {}).get("Phase", ["N/A"])[0]  # Assuming first phase is primary
                    enrollment = design_module.get("EnrollmentInfo", {}).get("EnrollmentCount", "N/A")

                    # Extracting location info
                    location_list = contacts_module.get("LocationList", {}).get("Location", [])
                    location_cities = [location.get("LocationCity", "N/A") for location in location_list if location.get("LocationCountry") in countries]
                    location_cities_str = ", ".join(location_cities)

                    # Add each study's details along with DesignPrimaryPurpose to the list
                    study_details_with_purpose.append({
                        "NCT ID": nct_id,
                        "DesignPrimaryPurpose": design_primary_purpose,
                        "Title": title,
                        "Status": status,
                        "Start Date": start_date,
                        "Sponsor": sponsor,
                        "Collaborator": collaborator,
                        "Description": description,
                        "Phase": phase,
                        "Enrollment": enrollment,
                        "Location Cities": location_cities_str,
                        "Overall Exclusion": compliance,
                        "Exclusion Trust Score": compliance_trust_score,
                        "Inclusion Probability": inclusion_probability,
                        "Inclusion Trust Score": inclusion_trust_score
                    })

    # Call the response function to get the sorted and formatted output
    final_response_json = response(study_details_with_purpose)
    return final_response_json

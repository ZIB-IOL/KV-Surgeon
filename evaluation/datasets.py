"""Dataset loaders for QuALITY, LongHealth and Long Code Arena (lca).

Every loader returns a list of dicts with ``article_id``, ``title``, ``article``
and ``questions``. Multiple-choice questions have ``question``, ``options``,
``gold_label`` (1-indexed) and ``question_unique_id``. String-match questions
(lca) have ``answers``, ``answer_prefix``, ``max_new_tokens`` and ``task`` instead
of ``options``/``gold_label``.
"""
import json
import urllib.request
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).parent.parent / "data"

# The LongHealth authors ask not to share examples publicly, so the file is
# fetched from their repository instead of being shipped here.
LONGHEALTH_URL = "https://raw.githubusercontent.com/kbressem/LongHealth/main/data/benchmark_v5.json"

def load_quality_data(data_path: str) -> List[Dict]:
    """Load QuALITY from JSONL, merging entries that share an article."""
    print(f"Loading QuALITY data from: {data_path}")
    combined_articles = {}
    total_entries = 0

    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                total_entries += 1
                article = json.loads(line)
                original_article_id = article['article_id']
                article['article'] = article.get('article', '').strip()
                key = (original_article_id, article['article'])

                questions = article.get('questions', [])

                article['article_id'] = f"quality_{original_article_id}"
                article['questions'] = questions

                if key in combined_articles:
                    combined_articles[key]['questions'].extend(questions)
                else:
                    combined_articles[key] = article

    data = list(combined_articles.values())
    print(f"Loaded {len(data)} unique articles (from {total_entries} entries)")
    return data


def load_longhealth_data(
    data_path: str,
    include_diagnosis: bool = True,
    patients_per_article: int = 1
) -> List[Dict]:
    """Load LongHealth, grouping all patients (sorted by ID) into articles of ``patients_per_article``.

    With more than one patient per article, each question is prefixed with the
    patient's info (including the diagnosis if ``include_diagnosis``).
    """
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"Downloading LongHealth from {LONGHEALTH_URL}")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LONGHEALTH_URL, data_path)
    print(f"Loading LongHealth data from: {data_path}")

    with open(data_path, 'r') as f:
        raw_data = json.load(f)

    sorted_patient_ids = sorted(raw_data.keys())

    num_articles = len(sorted_patient_ids) // patients_per_article
    data = []

    for article_idx in range(num_articles):
        start_idx = article_idx * patients_per_article
        end_idx = start_idx + patients_per_article
        group_patient_ids = sorted_patient_ids[start_idx:end_idx]

        article_parts = []
        all_questions = []
        patient_infos = []

        for patient_id in group_patient_ids:
            patient_data = raw_data[patient_id]

            texts = patient_data['texts']
            for note_id, note_text in texts.items():
                article_parts.append(f"<{note_id}>\n{note_text}\n</{note_id}>")

            if include_diagnosis:
                patient_info = f"ID {patient_id}, Name: {patient_data['name']}, Birthday: {patient_data['birthday']}, Diagnosis: {patient_data['diagnosis']}"
            else:
                patient_info = f"ID {patient_id}, Name: {patient_data['name']}, Birthday: {patient_data['birthday']}"

            patient_infos.append((patient_id, patient_info, patient_data))

        article = "\n\n".join(article_parts).strip()

        for patient_id, patient_info, patient_data in patient_infos:
            for q_idx, q in enumerate(patient_data['questions']):
                options = [
                    q['answer_a'],
                    q['answer_b'],
                    q['answer_c'],
                    q['answer_d'],
                    q['answer_e']
                ]

                correct_answer = q['correct']
                gold_label_idx = None

                for i, opt in enumerate(options):
                    if opt.strip() == correct_answer.strip():
                        gold_label_idx = i + 1
                        break

                if gold_label_idx is None:
                    print(f"Warning: Could not match correct answer '{correct_answer}' to options for {patient_id}, Q{q_idx}")
                    gold_label_idx = 1

                if patients_per_article > 1:
                    prefixed_question = (
                        f"Please answer the question below about the following patient: "
                        f"{patient_info}\n\n{q['question']}"
                    )
                else:
                    prefixed_question = q['question']

                all_questions.append({
                    'question': prefixed_question,
                    'options': options,
                    'gold_label': gold_label_idx,
                    'question_unique_id': f"{patient_id}_q{q.get('No', q_idx)}",
                })

        if patients_per_article == 1:
            patient_id = group_patient_ids[0]
            patient_data = raw_data[patient_id]
            article_entry = {
                'article_id': f"longhealth_{patient_id}",
                'title': patient_data['name'],
                'article': article,
                'questions': all_questions,
            }
        else:
            article_id = f"longhealth_group_{article_idx:02d}_patients_{start_idx+1:02d}-{end_idx:02d}"
            title = f"Patients {start_idx+1}-{end_idx}"
            article_entry = {
                'article_id': article_id,
                'title': title,
                'article': article,
                'questions': all_questions,
            }

        data.append(article_entry)

    print(f"Loaded {len(data)} grouped articles ({patients_per_article} patients per article)")
    return data


def load_lca_data() -> List[Dict]:
    """Load Long Code Arena project-level code completion as repository contexts with many line completions.

    ``data/lca_repo_completion.json`` is built by ``scripts/build_lca_repo_completion.py``.
    Each article is one repository snapshot (``.py`` files joined with
    ``[start of <path>]`` markers, so use ``chunking: lqa_pack``). Each question
    shows a file prefix and asks for the next line. ``task`` is ``lca_<class>``
    (infile, inproject, commited, common, non_informative, random).
    """
    data_path = DATA_DIR / "lca_repo_completion.json"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found; build it with scripts/build_lca_repo_completion.py"
        )
    with open(data_path) as f:
        payload = json.load(f)
    data = payload['articles']
    for article in data:
        for q in article['questions']:
            if 'ruler_outputs' in q:
                q['answers'] = q.pop('ruler_outputs')
    total_questions = sum(len(a['questions']) for a in data)
    print(f"Loaded {len(data)} lca repositories with {total_questions} completion questions "
          f"(median {payload['meta'].get('qwen_tokens_median', '?')} Qwen tokens per context)")
    return data


DATASET_PATHS = {
    'quality': DATA_DIR / 'QuALITY.v1.0.1.htmlstripped.dev',
    'longhealth': DATA_DIR / 'longhealth_benchmark_v5.json',
}


def load_dataset(dataset_name: str, include_diagnosis: bool = True) -> List[Dict]:
    """Load 'quality', 'longhealth', 'longhealthX' (X patients per article, e.g. 'longhealth5') or 'lca'."""
    if dataset_name == 'quality':
        return load_quality_data(DATASET_PATHS['quality'])

    elif dataset_name.startswith('longhealth'):
        patients_per_article = 1 if dataset_name == 'longhealth' else int(dataset_name[len('longhealth'):])

        return load_longhealth_data(
            DATASET_PATHS['longhealth'],
            include_diagnosis=include_diagnosis,
            patients_per_article=patients_per_article
        )

    elif dataset_name == 'lca':
        return load_lca_data()

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: 'quality', 'longhealth', 'longhealthX' (e.g., 'longhealth5'), 'lca'"
        )


# Datasets scored by string match against `answers` instead of multiple-choice
# parsing; answers are generated with thinking off.
STRING_MATCH_DATASET_PREFIXES = ('lca',)


def is_string_match_dataset(dataset_name: str) -> bool:
    """True if the dataset is scored by string match rather than multiple choice."""
    return dataset_name.startswith(STRING_MATCH_DATASET_PREFIXES)

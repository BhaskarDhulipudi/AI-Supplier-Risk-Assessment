from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import json
import secrets
from dotenv import load_dotenv
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv()

import math
import os
import re
import urllib.error
import urllib.request
from openai import OpenAI
from utils.document_processing import save_uploaded_file, extract_text_from_pdf, chunk_text
from supplier_data import load_suppliers, find_supplier, supplier_profile_text, calculate_risk_breakdown, calculate_overall_risk, score_to_label

app = Flask(__name__)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv('DATA_DIR', PROJECT_DIR)
PRODUCTION = (
    os.getenv('APP_ENV', '').lower() == 'production'
    or os.getenv('RENDER', '').lower() == 'true'
)
secret_key = os.getenv('SECRET_KEY')
if PRODUCTION and not secret_key:
    raise RuntimeError('SECRET_KEY must be set when running in production.')

os.makedirs(DATA_DIR, exist_ok=True)
app.secret_key = secret_key or secrets.token_hex(32)
app.config['UPLOAD_FOLDER'] = os.path.join(DATA_DIR, 'uploads')
app.config['USERS_FILE'] = os.path.join(DATA_DIR, 'users.json')
app.config['VECTOR_STORE_FILE'] = os.path.join(DATA_DIR, 'vector_store.json')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['OPENAI_MODEL'] = os.getenv('OPENAI_MODEL', 'gpt-6-luna')
app.config['OPENAI_EMBEDDING_MODEL'] = 'text-embedding-3-small'
app.config['OLLAMA_BASE_URL'] = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
app.config['OLLAMA_MODEL'] = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = PRODUCTION
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def load_users():
    if not os.path.exists(app.config['USERS_FILE']):
        if PRODUCTION:
            username = os.getenv('INITIAL_ADMIN_USERNAME', '').strip()
            password = os.getenv('INITIAL_ADMIN_PASSWORD', '')
            if not username or not password:
                raise RuntimeError(
                    'Set INITIAL_ADMIN_USERNAME and INITIAL_ADMIN_PASSWORD before '
                    'the first production login.'
                )
            save_users({username: generate_password_hash(password)})
        else:
            save_users({'admin': 'admin'})
    with open(app.config['USERS_FILE'], 'r', encoding='utf-8') as f:
        return json.load(f)


def save_users(users):
    with open(app.config['USERS_FILE'], 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2)


def verify_user_password(stored_password, password):
    if stored_password.startswith(('scrypt:', 'pbkdf2:')):
        return check_password_hash(stored_password, password), False
    return stored_password == password, stored_password == password


def load_vector_store():
    if not os.path.exists(app.config['VECTOR_STORE_FILE']):
        save_vector_store([])
    with open(app.config['VECTOR_STORE_FILE'], 'r', encoding='utf-8') as f:
        return json.load(f)


def save_vector_store(store):
    with open(app.config['VECTOR_STORE_FILE'], 'w', encoding='utf-8') as f:
        json.dump(store, f, indent=2)


def get_openai_api_key():
    return os.getenv('OPENAI_API_KEY')


def has_openai_key():
    return bool(get_openai_api_key())


def get_openai_client():
    key = get_openai_api_key()

    if not key:
        raise RuntimeError(
            'OPENAI_API_KEY environment variable is not set.'
        )

    return OpenAI(api_key=key)


def generate_ollama_response(prompt):
    request_body = json.dumps({
        'model': app.config['OLLAMA_MODEL'],
        'prompt': prompt,
        'stream': False,
        'format': 'json',
        'options': {'temperature': 0.1},
    }).encode('utf-8')
    request = urllib.request.Request(
        f"{app.config['OLLAMA_BASE_URL'].rstrip('/')}/api/generate",
        data=request_body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=180) as response:
        payload = json.loads(response.read().decode('utf-8'))
    assistant_output = payload.get('response')
    if not assistant_output:
        raise ValueError('Ollama returned an empty response.')
    return assistant_output


def merge_model_assessment(result, assistant_output):
    parsed = json.loads(assistant_output)
    if not isinstance(parsed, dict):
        raise ValueError('The model response must be a JSON object.')

    required_text = ('risk_category', 'risk_level', 'summary', 'procurement_action')
    if any(not isinstance(parsed.get(key), str) or not parsed[key].strip() for key in required_text):
        raise ValueError('The model response is missing required text fields.')
    if parsed['risk_level'] not in {'Low', 'Medium', 'High', 'Critical'}:
        raise ValueError('The model returned an invalid risk level.')
    if parsed['procurement_action'] not in {
        'Approve', 'Approve with Monitoring', 'Review', 'Hold', 'Reject'
    }:
        raise ValueError('The model returned an invalid procurement action.')

    score = parsed.get('risk_score')
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError('The model risk score must be a number from 0 to 100.')
    for key in ('key_risks', 'evidence'):
        if not isinstance(parsed.get(key), list) or any(not isinstance(item, str) for item in parsed[key]):
            raise ValueError(f'The model response contains an invalid {key} list.')

    sources = parsed.get('data_sources')
    if isinstance(sources, dict):
        normalized_sources = []
        for source, details in sources.items():
            if not isinstance(source, str):
                raise ValueError('The model response contains an invalid data source.')
            if isinstance(details, list) and all(isinstance(item, str) for item in details):
                detail_text = ', '.join(details)
            elif isinstance(details, str):
                detail_text = details
            else:
                raise ValueError('The model response contains an invalid data source detail.')
            normalized_sources.append(f'{source}: {detail_text}' if detail_text else source)
        sources = normalized_sources
    elif not isinstance(sources, list) or any(not isinstance(item, str) for item in sources):
        raise ValueError('The model response contains an invalid data_sources list.')

    result.update({
        'risk_category': parsed['risk_category'],
        'summary': parsed['summary'],
        'key_risks': parsed['key_risks'],
        'evidence': parsed['evidence'],
        'data_sources': sources,
    })
    result['summary'] += (
        f" Overall supplier assessment remains {result['risk_level']} "
        f"(score {result['risk_score']}) with the recommended action: "
        f"{result['procurement_action']}."
    )
    return result


def get_embedding(text):
    client = get_openai_client()

    response = client.embeddings.create(
        model=app.config['OPENAI_EMBEDDING_MODEL'],
        input=text
    )

    return response.data[0].embedding


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-10)


def retrieve_relevant_chunks(query, k=4, store=None):
    store = load_vector_store() if store is None else store
    if not store:
        return []
    if has_openai_key() and any(item.get('embedding') for item in store):
        query_embedding = get_embedding(query)
        scored = []
        for item in store:
            if item.get('embedding') is None:
                continue
            score = cosine_similarity(query_embedding, item['embedding'])
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item['chunk_text'] for score, item in scored[:k]]

    query_tokens = set(re.findall(r"\w+", query.lower()))
    scored = []
    for item in store:
        text = item['chunk_text'].lower()
        score = sum(1 for token in query_tokens if token in text)
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item['chunk_text'] for score, item in scored[:k] if score > 0]


def get_document_text(file_path):
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def ingest_document(file_path, source):
    text = get_document_text(file_path)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError('No readable text was found in the document.')
    store = load_vector_store()
    source_file = os.path.relpath(file_path, start=os.getcwd())
    existing_source_files = {item.get('source_file') for item in store}
    if source_file in existing_source_files:
        return
    for index, chunk in enumerate(chunks):
        embedding = None
        if has_openai_key():
            try:
                embedding = get_embedding(chunk)
            except Exception:
                embedding = None
        store.append({
            'id': f'{source}-{index}-{len(store)}',
            'source': source,
            'source_file': source_file,
            'chunk_text': chunk,
            'embedding': embedding
        })
    save_vector_store(store)


def save_and_ingest_uploaded_document(uploaded_file, supplier_name, supplier=None):
    file_path = save_uploaded_file(uploaded_file, app.config['UPLOAD_FOLDER'])
    source = supplier_name or (supplier or {}).get('Supplier_Name') or os.path.basename(uploaded_file.filename)
    try:
        ingest_document(file_path, source)
    except Exception:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        raise
    return file_path


def get_vector_store_metadata():
    store = load_vector_store()
    sources = {}
    for item in store:
        source = item.get('source', 'Unknown')
        sources[source] = sources.get(source, 0) + 1
    return {
        'total_chunks': len(store),
        'source_counts': sorted(sources.items(), key=lambda x: x[0])
    }


def ingest_existing_uploads():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    files_to_ingest = []
    for root, _, files in os.walk(app.config['UPLOAD_FOLDER']):
        for filename in files:
            if filename.lower().endswith(('.pdf', '.txt')):
                files_to_ingest.append(os.path.join(root, filename))

    store = load_vector_store()
    existing_source_files = {item.get('source_file') for item in store}
    ingested = 0
    skipped = 0
    for file_path in files_to_ingest:
        rel_path = os.path.relpath(file_path, start=os.getcwd())
        if rel_path in existing_source_files:
            skipped += 1
            continue
        source = os.path.splitext(os.path.basename(file_path))[0]
        ingest_document(file_path, source)
        ingested += 1
    return ingested, skipped


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapped_view

@app.context_processor
def inject_user():
    return {
        'current_user': session.get('user'),
        'is_admin': session.get('user') == 'admin'
    }

@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    users = load_users()
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            return render_template('login.html', error='Please enter both username and password.')
        stored_password = users.get(username)
        if stored_password is None:
            return render_template('login.html', error='Invalid username or password.')
        password_matches, needs_upgrade = verify_user_password(stored_password, password)
        if not password_matches:
            return render_template('login.html', error='Invalid username or password.')
        if needs_upgrade:
            users[username] = generate_password_hash(password)
            save_users(users)
        session['user'] = username
        return redirect(url_for('dashboard'))
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', active_page='dashboard')

@app.route('/upload')
@login_required
def upload():
    return render_template(
        'upload.html',
        active_page='upload',
        message=request.args.get('message', ''),
        error=request.args.get('error', '')
    )

@app.route('/suppliers')
@login_required
def suppliers():
    return render_template('suppliers.html', active_page='suppliers')

@app.route('/assistant')
@login_required
def assistant():
    return render_template('assistant.html', active_page='assistant')

@app.route('/reports')
@login_required
def reports():
    return render_template('reports.html', active_page='reports')

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    users = load_users()
    message = request.args.get('message', '')
    error = request.args.get('error', '')
    if request.method == 'POST':
        if session.get('user') != 'admin':
            error = 'Only admin can manage users.'
        else:
            if request.form.get('delete_username'):
                delete_username = request.form.get('delete_username')
                if delete_username in users and delete_username != 'admin':
                    users.pop(delete_username)
                    save_users(users)
                    message = f'User "{delete_username}" removed successfully.'
                else:
                    error = 'Cannot remove admin or unknown user.'
            else:
                new_username = request.form.get('new_username', '').strip()
                new_password = request.form.get('new_password', '')
                if not new_username or not new_password:
                    error = 'Enter both a username and password.'
                elif new_username in users:
                    error = 'That username already exists.'
                else:
                    users[new_username] = generate_password_hash(new_password)
                    save_users(users)
                    message = 'New user added successfully.'
    vector_metadata = get_vector_store_metadata()
    if session.get('user') != 'admin':
        users = {}
    return render_template(
        'settings.html',
        active_page='settings',
        users=users,
        message=message,
        error=error,
        vector_metadata=vector_metadata,
        ollama_model=app.config['OLLAMA_MODEL'],
        openai_enabled=has_openai_key()
    )

@app.route('/ingest-existing-uploads', methods=['POST'])
@login_required
def ingest_existing_uploads_route():
    if session.get('user') != 'admin':
        return redirect(url_for('settings', error='Only admin can ingest documents.'))
    try:
        ingested, skipped = ingest_existing_uploads()
        message = f'Ingested {ingested} files, skipped {skipped} already ingested.'
        return redirect(url_for('settings', message=message))
    except Exception as exc:
        return redirect(url_for('settings', error=f'Ingestion failed: {exc}'))

@app.route('/vector-store')
@login_required
def vector_store():
    store = load_vector_store()
    return render_template('vector_store.html', active_page='vector_store', store=store)

@app.route('/upload-document', methods=['POST'])
@login_required
def upload_document():
    supplier_name = request.form.get('supplier_name', '').strip()
    uploaded_file = request.files.get('document')
    if not uploaded_file or not uploaded_file.filename:
        return render_template(
            'upload.html',
            active_page='upload',
            error='Choose a PDF or TXT document to upload.'
        ), 400

    try:
        save_and_ingest_uploaded_document(uploaded_file, supplier_name)
    except (ValueError, RuntimeError) as exc:
        return render_template('upload.html', active_page='upload', error=str(exc)), 400
    except Exception as exc:
        return render_template(
            'upload.html',
            active_page='upload',
            error=f'Document upload failed: {exc}'
        ), 500

    return redirect(url_for('upload', message='Document uploaded and processed successfully.'))

@app.route('/api/suppliers')
@login_required
def api_suppliers():
    rows = load_suppliers()
    suppliers = []
    for r in rows:
        breakdown = calculate_risk_breakdown(r)
        suppliers.append({
            'Supplier_ID': r.get('Supplier_ID'),
            'Supplier_Name': r.get('Supplier_Name'),
            'Industry': r.get('Industry'),
            'Country': r.get('Country'),
            'Financial_Risk_Score': breakdown['financial_risk_score'],
            'Financial_Risk_Level': breakdown['financial_risk_level'],
            'Product_Risk_Score': breakdown['product_risk_score'],
            'Product_Risk_Level': breakdown['product_risk_level'],
            'Quality_Risk_Score': breakdown['quality_risk_score'],
            'Quality_Risk_Level': breakdown['quality_risk_level'],
            'Overall_Risk_Score': breakdown['overall_risk_score'],
            'Overall_Risk_Level': breakdown['overall_risk_level'],
            'Risk_Level': breakdown['overall_risk_level'],
            'Risk_Reason': r.get('Risk_Reason'),
            'Recommended_Action': r.get('Recommended_Action') or 'Monitor supplier'
        })
    return jsonify({'count': len(suppliers), 'suppliers': suppliers})


@app.route('/api/reports')
@login_required
def api_reports():
    rows = load_suppliers()
    reports = []
    ranked_suppliers = sorted(
        rows,
        key=lambda supplier: calculate_risk_breakdown(supplier)['overall_risk_score'],
        reverse=True
    )[:10]
    for supplier in ranked_suppliers:
        breakdown = calculate_risk_breakdown(supplier)
        reports.append({
            'supplier_id': supplier.get('Supplier_ID'),
            'supplier': supplier.get('Supplier_Name', 'Unknown Supplier'),
            'risk_reason': supplier.get('Risk_Reason'),
            'recommended_action': supplier.get('Recommended_Action') or 'Monitor supplier',
            'risk_level': breakdown['overall_risk_level'],
            'risk_score': breakdown['overall_risk_score'],
            'financial_risk_score': breakdown['financial_risk_score'],
            'financial_risk_level': breakdown['financial_risk_level'],
            'product_risk_score': breakdown['product_risk_score'],
            'product_risk_level': breakdown['product_risk_level'],
            'quality_risk_score': breakdown['quality_risk_score'],
            'quality_risk_level': breakdown['quality_risk_level'],
            'overall_risk_score': breakdown['overall_risk_score'],
            'overall_risk_level': breakdown['overall_risk_level']
        })
    return jsonify({'count': len(reports), 'reports': reports})


@app.route('/api/supplier/<supplier_id>')
@login_required
def api_supplier(supplier_id):
    supplier = find_supplier(supplier_id)
    if not supplier:
        return jsonify({'error': 'Supplier not found'}), 404
    return jsonify({key: value for key, value in supplier.items() if key != '_risk_breakdown'})


def build_supplier_query(supplier, user_query):
    profile = supplier_profile_text(supplier)
    supplier_name = supplier.get('Supplier_Name', '') if supplier else ''
    return f"Supplier: {supplier_name}\nUser question: {user_query}\nStructured supplier profile:\n{profile}"


def is_supplier_risk_question(query):
    relevant_terms = (
        r'\bsuppliers?\b',
        r'\bvendors?\b',
        r'\bprocurement\b',
        r'\bsourc(?:e|ing)\b',
        r'\bpurchas(?:e|ing)\b',
        r'\b(?:purchase\s+orders?|pos?)\b',
        r'\brisks?\b',
        r'\bfinancial\b',
        r'\bcredit\b',
        r'\bdebt\b',
        r'\b(?:cash\s+flow|revenue|liquidity|insolvency|bankruptcy)\b',
        r'\bgeopolit(?:ical|ics)\b',
        r'\bpolitic(?:al|s)\b',
        r'\bcountr(?:y|ies)\b',
        r'\bexpos(?:ure|ures|ed)\b',
        r'\boperational\b',
        r'\bdeliver(?:y|ies|ed|ing)?\b',
        r'\bshipping\b',
        r'\bsupply\s+chain\b',
        r'\bcontinuity\b',
        r'\bquality\b',
        r'\bdefects?\b',
        r'\bcompliance\b',
        r'\bregulatory\b',
        r'\baudits?\b',
        r'\bcyber(?:security)?\b',
        r'\bsecurity\b',
        r'\besg\b',
        r'\bsustainab(?:ility|le)\b',
        r'\benvironmental\b',
        r'\bcontracts?\b',
        r'\blead\s+time\b',
        r'\bdependenc(?:y|ies)\b',
        r'\bscor(?:e|es|ing)\b',
        r'\bassess(?:ment|ments)?\b',
        r'\bmitigat(?:e|ion|ions|ing)\b',
        r'\bapprov(?:e|al|als|ed|ing)\b',
        r'\breject(?:ed|ion|ions)?\b',
        r'\bhold\b',
        r'\bmonitor(?:ing)?\b',
        r'\brecommend(?:ation|ations|ed)?\b',
        r'\bevidence\b',
        r'\bsupplier\s+data\b',
        r'\bdataset\b',
        r'\binsurance\b',
        r'\b(?:product\s+recalls?|business\s+interruption)\b',
    )
    return any(re.search(term, query, re.IGNORECASE) for term in relevant_terms)


def retrieve_supplier_evidence(supplier, query, k=4):
    if not supplier:
        return retrieve_relevant_chunks(query, k=k)

    supplier_keys = {
        re.sub(r"\W+", "", str(value).lower())
        for value in (supplier.get('Supplier_ID'), supplier.get('Supplier_Name'))
        if value
    }
    supplier_documents = []
    for item in load_vector_store():
        source = re.sub(r"\W+", "", str(item.get('source', '')).lower())
        filename = os.path.splitext(os.path.basename(item.get('source_file', '')))[0]
        filename = re.sub(r"\W+", "", filename.lower())
        if source in supplier_keys or filename in supplier_keys:
            supplier_documents.append(item)

    return retrieve_relevant_chunks(build_supplier_query(supplier, query), k=k, store=supplier_documents)


def local_risk_assessment(supplier, query, relevant_chunks):
    supplier_name = (supplier or {}).get('Supplier_Name', 'Unknown Supplier')
    supplier_id = (supplier or {}).get('Supplier_ID')
    breakdown = calculate_risk_breakdown(supplier or {})
    document_text = ' '.join(relevant_chunks).lower()
    risk_signal_groups = {
        'financial': ('financial_risk_score', ('bankruptcy', 'insolvency', 'payment default', 'credit downgrade', 'liquidity shortfall', 'cash flow shortfall')),
        'product_delivery': ('product_risk_score', ('late delivery', 'delivery delay', 'delivery failure', 'supply disruption', 'production stoppage', 'production halt', 'shortage')),
        'quality_compliance': ('quality_risk_score', ('defect', 'failed audit', 'noncompliance', 'non-compliance', 'regulatory violation', 'product recall', 'quality control gap', 'safety incident'))
    }
    detected_signals = []
    for category, (score_key, phrases) in risk_signal_groups.items():
        if any(phrase in document_text for phrase in phrases):
            breakdown[score_key] = min(100, breakdown[score_key] + 8)
            breakdown[score_key.replace('_score', '_level')] = score_to_label(breakdown[score_key])
            detected_signals.append(category)

    breakdown['overall_risk_score'] = calculate_overall_risk(
        breakdown['financial_risk_score'],
        breakdown['product_risk_score'],
        breakdown['quality_risk_score']
    )
    breakdown['overall_risk_level'] = score_to_label(breakdown['overall_risk_score'])
    score = breakdown['overall_risk_score']
    risk_level = score_to_label(score)

    if risk_level == 'Critical':
        procurement_action = 'Reject'
    elif risk_level == 'High':
        procurement_action = 'Hold'
    elif risk_level == 'Medium':
        procurement_action = 'Review'
    elif risk_level == 'Low':
        procurement_action = 'Approve with Monitoring'

    evidence = relevant_chunks[:3] if relevant_chunks else [
        f"Supplier profile indicates a {risk_level.lower()} risk baseline for {supplier_name}.",
        f"The CSV dataset includes supplier risk indicators for {supplier_name}.",
    ]

    key_risks = []
    if supplier and supplier.get('Risk_Reason'):
        key_risks.append(str(supplier['Risk_Reason']))
    signal_labels = {
        'financial': 'Financial distress signals in supplier documents',
        'product_delivery': 'Delivery or supply continuity signals in supplier documents',
        'quality_compliance': 'Quality or compliance signals in supplier documents'
    }
    key_risks.extend(signal_labels[signal] for signal in detected_signals)
    if not key_risks:
        key_risks = ['Supplier risk signals require additional review.']

    summary = (
        f"Local assessment for {supplier_name} indicates {risk_level} risk. "
        f"The evidence and supplier profile suggest {', '.join(key_risks[:2]) if key_risks else 'elevated operational concern'}."
    )
    geopolitical_score = (supplier or {}).get('Geopolitical_Risk_Score')
    if geopolitical_score not in (None, ''):
        geopolitical_score = float(geopolitical_score)
        summary += (
            f" The CSV also reports a geopolitical risk score of {geopolitical_score:g} "
            f"({score_to_label(geopolitical_score)}); this factor is not included in the overall score."
        )

    return {
        'supplier': supplier_name,
        'supplier_id': supplier_id,
        'query': query or '',
        'risk_category': 'Supplier Risk',
        'risk_level': risk_level,
        'risk_score': score,
        'risk_breakdown': breakdown,
        'summary': summary,
        'procurement_action': procurement_action,
        'key_risks': key_risks[:4],
        'evidence': evidence[:3],
        'data_sources': ['CSV supplier dataset'] + (['Supplier-specific RAG documents'] if relevant_chunks else [])
    }


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    supplier_name = request.form.get('supplier_name', '').strip()
    supplier_id = request.form.get('supplier_id', '').strip()
    query = request.form.get('query', '').strip()
    uploaded_file = request.files.get('document')

    if not query:
        return jsonify({'error': 'Please enter a question for the assistant.'}), 400

    if not is_supplier_risk_question(query):
        return jsonify({
            'error': 'I can only help with supplier risk, procurement, and related assessment questions.'
        }), 422

    supplier = find_supplier(supplier_id or supplier_name)

    if uploaded_file and uploaded_file.filename:
        try:
            save_and_ingest_uploaded_document(uploaded_file, supplier_name, supplier)
        except (ValueError, RuntimeError) as exc:
            return jsonify({'error': str(exc)}), 400
        except Exception as exc:
            return jsonify({'error': f'Document ingestion failed: {str(exc)}'}), 500

    try:
        relevant_chunks = retrieve_supplier_evidence(supplier, query, k=5)
    except Exception as exc:
        return jsonify({'error': f'Retrieval failed: {str(exc)}'}), 500

    structured_context = supplier_profile_text(supplier)
    document_context = '\n\n'.join(relevant_chunks) if relevant_chunks else 'No relevant document excerpts were available.'
    result = local_risk_assessment(supplier, query, relevant_chunks)

    prompt = f'''You are an AI supplier risk analyst supporting procurement decisions.

Use BOTH sources below:
A) STRUCTURED SUPPLIER DATA from the project CSV.
B) RETRIEVED DOCUMENT EVIDENCE from the project's RAG vector store.

Never invent supplier facts. If a fact is not present in A or B, state that it is unavailable.
The server-calculated overall score, risk level, and procurement action below are authoritative. Preserve them exactly in the JSON. Do not substitute a component score, such as geopolitical risk, for the overall score. You may explain component risks in the summary and key risks.
Provide a procurement-oriented decision.
Answer the user's specific question directly in the summary before giving supporting risk context.

Return valid JSON with exactly these keys:
"risk_category", "risk_level", "risk_score", "summary", "procurement_action", "key_risks", "evidence", "data_sources"

Rules:
- risk_score must be a number from 0 to 100.
- risk_level must be one of Low, Medium, High, Critical.
- Use these score bands: Low 0-34, Medium 35-54, High 55-74, Critical 75-100.
- procurement_action should be one of Approve, Approve with Monitoring, Review, Hold, Reject.
- Match the action to the risk level: Low = Approve with Monitoring, Medium = Review, High = Hold, Critical = Reject.
- key_risks must be an array of short strings.
- evidence must be an array of short statements grounded in the provided sources.
- data_sources must be an array of strings identifying which conclusions came from CSV and which came from RAG documents.

STRUCTURED SUPPLIER DATA:
{structured_context}

RAG DOCUMENT EVIDENCE:
{document_context}

AUTHORITATIVE OVERALL ASSESSMENT:
Risk score: {result['risk_score']}
Risk level: {result['risk_level']}
Procurement action: {result['procurement_action']}

USER QUESTION:
{query}
'''

    if has_openai_key():
        provider = 'OpenAI'
        try:
            client = get_openai_client()
            response = client.chat.completions.create(
                model=app.config['OPENAI_MODEL'],
                messages=[
                    {'role': 'system', 'content': 'You produce evidence-grounded supplier risk assessments for procurement teams.'},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.1,
                max_tokens=900,
            )
            assistant_output = response.choices[0].message.content.strip()
        except Exception as exc:
            result['summary'] = f"OpenAI request failed: {exc}. Returning the offline rule-based assessment. " + result['summary']
            return jsonify(result)
    else:
        provider = 'Ollama'
        try:
            assistant_output = generate_ollama_response(prompt)
        except urllib.error.HTTPError as exc:
            error_details = exc.read().decode('utf-8', errors='replace')
            if exc.code == 404 and 'model' in error_details.lower():
                reason = (
                    f"Model {app.config['OLLAMA_MODEL']} is not downloaded. "
                    f"Run `ollama pull {app.config['OLLAMA_MODEL']}` and try again."
                )
            else:
                reason = f"Ollama returned HTTP {exc.code}: {error_details or exc.reason}"
            result['summary'] = (
                f"{reason} Returning the offline rule-based assessment. "
                + result['summary']
            )
            return jsonify(result)
        except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
            result['summary'] = (
                f"Ollama is unavailable or could not generate a valid response using "
                f"{app.config['OLLAMA_MODEL']}: {exc}. Returning the offline rule-based assessment. "
                + result['summary']
            )
            return jsonify(result)

    try:
        result = merge_model_assessment(result, assistant_output)
    except (json.JSONDecodeError, ValueError) as exc:
        result['summary'] = (
            f"{provider} returned an invalid assessment response: {exc}. "
            f"Returning the offline rule-based assessment. {result['summary']}"
        )
        return jsonify(result)

    return jsonify(result)

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', '5000')),
        debug=os.getenv('FLASK_DEBUG') == '1'
    )
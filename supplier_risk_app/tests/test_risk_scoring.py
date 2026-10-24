import unittest
import json
import os
import re
import tempfile
import urllib.error
from io import BytesIO
from unittest.mock import Mock, patch

from app import app, generate_ollama_response, is_supplier_risk_question, load_users, local_risk_assessment, merge_model_assessment, retrieve_relevant_chunks, retrieve_supplier_evidence, verify_user_password
from supplier_data import calculate_overall_risk, calculate_risk_breakdown, load_suppliers


class RiskScoringTests(unittest.TestCase):
    def test_risk_breakdown_has_all_four_scores(self):
        supplier = load_suppliers()[0]
        breakdown = calculate_risk_breakdown(supplier)

        self.assertIn('financial_risk_score', breakdown)
        self.assertIn('product_risk_score', breakdown)
        self.assertIn('quality_risk_score', breakdown)
        self.assertIn('overall_risk_score', breakdown)
        self.assertIn('overall_risk_level', breakdown)

        self.assertTrue(0 <= breakdown['financial_risk_score'] <= 100)
        self.assertTrue(0 <= breakdown['product_risk_score'] <= 100)
        self.assertTrue(0 <= breakdown['quality_risk_score'] <= 100)
        self.assertTrue(0 <= breakdown['overall_risk_score'] <= 100)

    def test_loaded_supplier_breakdown_is_not_recalculated(self):
        supplier = next(row for row in load_suppliers() if row['Supplier_ID'] == 'SUP0678')

        first = calculate_risk_breakdown(supplier)
        second = calculate_risk_breakdown(supplier)

        self.assertEqual(first, second)
        self.assertEqual(first['financial_risk_score'], 97)

    def test_local_assessment_uses_weighted_baseline_and_consistent_band(self):
        supplier = next(row for row in load_suppliers() if row['Supplier_ID'] == 'SUP0678')

        result = local_risk_assessment(supplier, 'Assess every risk area.', [])

        self.assertEqual(result['risk_score'], calculate_risk_breakdown(supplier)['overall_risk_score'])
        self.assertEqual(result['risk_level'], 'High')
        self.assertEqual(result['procurement_action'], 'Hold')
        self.assertEqual(result['risk_breakdown']['overall_risk_level'], 'High')

    def test_local_assessment_discloses_geopolitical_score_excluded_from_overall(self):
        supplier = next(row for row in load_suppliers() if row['Supplier_ID'] == 'SUP0004')

        result = local_risk_assessment(supplier, 'Assess supplier risk.', [])

        self.assertEqual(result['risk_score'], 21)
        self.assertEqual(result['risk_level'], 'Low')
        self.assertIn('geopolitical risk score of 61.07 (High)', result['summary'])
        self.assertIn('not included in the overall score', result['summary'])

    def test_supplier_document_signals_raise_score_without_using_question_words(self):
        supplier = next(row for row in load_suppliers() if row['Supplier_ID'] == 'SUP0678')
        evidence = ['Recurring late delivery, production stoppage, and product defects were documented.']

        result = local_risk_assessment(supplier, 'No risk concerns.', evidence)
        baseline = calculate_risk_breakdown(supplier)
        expected_product = min(100, baseline['product_risk_score'] + 8)
        expected_quality = min(100, baseline['quality_risk_score'] + 8)
        expected_overall = calculate_overall_risk(
            baseline['financial_risk_score'], expected_product, expected_quality
        )

        self.assertEqual(result['risk_score'], expected_overall)
        self.assertEqual(result['risk_breakdown']['product_risk_score'], expected_product)
        self.assertEqual(result['risk_breakdown']['quality_risk_score'], expected_quality)
        self.assertEqual(result['risk_breakdown']['overall_risk_score'], result['risk_score'])
        self.assertEqual(result['risk_level'], 'High')
        self.assertEqual(result['procurement_action'], 'Hold')

    def test_supplier_evidence_does_not_mix_other_suppliers_documents(self):
        supplier = {'Supplier_ID': 'SUP0678', 'Supplier_Name': 'Supplier 0678'}
        store = [
            {'source': 'Supplier 0678', 'source_file': 'uploads/supplier-0678.txt', 'chunk_text': 'Supplier 0678 delivery and quality review.'},
            {'source': 'Other Supplier', 'source_file': 'uploads/other.txt', 'chunk_text': 'Supplier 0678 delivery quality risk findings.'}
        ]

        with patch('app.load_vector_store', return_value=store):
            evidence = retrieve_supplier_evidence(supplier, 'delivery quality risk')

        self.assertEqual(evidence, ['Supplier 0678 delivery and quality review.'])

    def test_keyword_retrieval_does_not_return_unmatched_document_chunks(self):
        store = [{'chunk_text': 'Contract renewal is due in December.'}]

        with patch('app.has_openai_key', return_value=False):
            evidence = retrieve_relevant_chunks('cybersecurity incident response', k=4, store=store)

        self.assertEqual(evidence, [])


class AssistantScopeTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        with self.client.session_transaction() as session:
            session['user'] = 'test'

    def test_unrelated_question_is_rejected_before_retrieval(self):
        with patch('app.retrieve_supplier_evidence') as retrieve:
            response = self.client.post(
                '/analyze',
                data={'query': 'What is the capital of France?', 'supplier_id': 'SUP0678'}
            )

        self.assertEqual(response.status_code, 422)
        self.assertIn('supplier risk', response.get_json()['error'])
        retrieve.assert_not_called()

    def test_supplier_risk_questions_are_in_scope(self):
        self.assertTrue(is_supplier_risk_question('Should we approve this supplier?'))
        self.assertTrue(is_supplier_risk_question('Explain its cybersecurity risk score.'))
        self.assertTrue(is_supplier_risk_question('How exposed is supplier 0004 to geopolitical tensions?'))
        self.assertFalse(is_supplier_risk_question('What is the capital of France?'))

    def test_model_response_validation_rejects_invalid_score_and_shape(self):
        base_result = {'summary': 'Local assessment.'}
        with self.assertRaisesRegex(ValueError, 'risk score'):
            merge_model_assessment(base_result, json.dumps({
                'risk_category': 'Supplier Risk',
                'risk_level': 'Low',
                'risk_score': 125,
                'summary': 'Assessment.',
                'procurement_action': 'Review',
                'key_risks': [],
                'evidence': [],
                'data_sources': [],
            }))

        with self.assertRaisesRegex(ValueError, 'key_risks list'):
            merge_model_assessment(base_result, json.dumps({
                'risk_category': 'Supplier Risk',
                'risk_level': 'Low',
                'risk_score': 20,
                'summary': 'Assessment.',
                'procurement_action': 'Approve with Monitoring',
                'key_risks': 'None',
                'evidence': [],
                'data_sources': [],
            }))

    def test_model_cannot_override_server_calculated_overall_assessment(self):
        base_result = {
            'summary': 'Local assessment.',
            'risk_level': 'Low',
            'risk_score': 21,
            'procurement_action': 'Approve with Monitoring',
        }
        merged = merge_model_assessment(base_result, json.dumps({
            'risk_category': 'Supplier Risk',
            'risk_level': 'High',
            'risk_score': 61,
            'summary': 'Assessment.',
            'procurement_action': 'Hold',
            'key_risks': [],
            'evidence': [],
            'data_sources': [],
        }))

        self.assertEqual(merged['risk_level'], 'Low')
        self.assertEqual(merged['risk_score'], 21)
        self.assertEqual(merged['procurement_action'], 'Approve with Monitoring')

    def test_model_data_source_mapping_is_normalized_to_readable_list(self):
        result = merge_model_assessment({
            'risk_score': 20,
            'risk_level': 'Low',
            'procurement_action': 'Approve with Monitoring',
        }, json.dumps({
            'risk_category': 'Geopolitical',
            'risk_level': 'High',
            'risk_score': 61,
            'summary': 'Assessment.',
            'procurement_action': 'Hold',
            'key_risks': [],
            'evidence': [],
            'data_sources': {
                'Structured Supplier Data': ['Geopolitical Risk Score', 'Country'],
                'RAG Document Evidence': [],
            },
        }))

        self.assertEqual(result['data_sources'], [
            'Structured Supplier Data: Geopolitical Risk Score, Country',
            'RAG Document Evidence',
        ])

    def test_no_api_key_returns_clearly_labeled_offline_assessment(self):
        with patch('app.has_openai_key', return_value=False), patch(
            'app.retrieve_supplier_evidence', return_value=[]
        ), patch('app.generate_ollama_response', side_effect=OSError('Ollama is not running')):
            response = self.client.post(
                '/analyze',
                data={'query': 'Assess supplier risk.', 'supplier_id': 'SUP0004'}
            )

        self.assertEqual(response.status_code, 200)
        summary = response.get_json()['summary']
        self.assertIn('Ollama is unavailable', summary)
        self.assertIn('Returning the offline rule-based assessment', summary)

    def test_ollama_response_is_used_without_openai_key(self):
        model_result = {
            'risk_category': 'Supplier Risk',
            'risk_level': 'Medium',
            'risk_score': 49,
            'summary': 'Ollama assessment based on the supplier record.',
            'procurement_action': 'Review',
            'key_risks': ['Supplier financial exposure'],
            'evidence': ['The supplier CSV contains financial risk data.'],
            'data_sources': ['CSV supplier dataset'],
        }
        with patch('app.has_openai_key', return_value=False), patch(
            'app.retrieve_supplier_evidence', return_value=[]
        ), patch('app.generate_ollama_response', return_value=json.dumps(model_result)) as generate:
            response = self.client.post(
                '/analyze',
                data={'query': 'Assess supplier risk.', 'supplier_id': 'SUP0004'}
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(model_result['summary'], response.get_json()['summary'])
        self.assertIn('Overall supplier assessment remains Low (score 21)', response.get_json()['summary'])
        self.assertEqual(response.get_json()['risk_score'], 21)
        self.assertEqual(response.get_json()['risk_level'], 'Low')
        self.assertEqual(response.get_json()['procurement_action'], 'Approve with Monitoring')
        generate.assert_called_once()

    def test_missing_ollama_model_returns_pull_instruction(self):
        missing_model = urllib.error.HTTPError(
            'http://localhost:11434/api/generate',
            404,
            'Not Found',
            {},
            Mock(read=Mock(return_value=b'{"error":"model not found"}')),
        )
        with patch('app.has_openai_key', return_value=False), patch(
            'app.retrieve_supplier_evidence', return_value=[]
        ), patch('app.generate_ollama_response', side_effect=missing_model):
            response = self.client.post(
                '/analyze',
                data={'query': 'Assess supplier risk.', 'supplier_id': 'SUP0004'}
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('ollama pull llama3.2:3b', response.get_json()['summary'])

    def test_ollama_request_uses_configured_model_and_json_output(self):
        mock_response = Mock()
        mock_response.__enter__ = lambda _: mock_response
        mock_response.__exit__ = lambda *_: None
        mock_response.read.return_value = json.dumps({'response': '{"summary": "ok"}'}).encode('utf-8')

        opener = Mock()
        opener.open.return_value = mock_response
        with patch('app.urllib.request.build_opener', return_value=opener):
            self.assertEqual(generate_ollama_response('assessment prompt'), '{"summary": "ok"}')

        request = opener.open.call_args.args[0]
        request_body = json.loads(request.data.decode('utf-8'))
        self.assertTrue(request.full_url.endswith('/api/generate'))
        self.assertEqual(request_body['model'], app.config['OLLAMA_MODEL'])
        self.assertEqual(request_body['prompt'], 'assessment prompt')
        self.assertEqual(request_body['format'], 'json')


class DashboardTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        with self.client.session_transaction() as session:
            session['user'] = 'dashboard-test'

    def test_dashboard_renders_data_loading_script(self):
        response = self.client.get('/dashboard')

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("fetch('/api/suppliers')", html)
        self.assertIn('Highest-Risk Suppliers', html)
        self.assertIn('Recommended Action', html)
        self.assertNotIn('Recent Supplier Assessments', html)

    def test_supplier_api_includes_risk_reason_for_dashboard(self):
        response = self.client.get('/api/suppliers')

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertGreater(data['count'], 0)
        self.assertIn('Risk_Reason', data['suppliers'][0])
        self.assertIn('Recommended_Action', data['suppliers'][0])


class ReportsPageTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        with self.client.session_transaction() as session:
            session['user'] = 'reports-test'

    def test_reports_api_returns_dataset_summaries_without_fake_report_metadata(self):
        response = self.client.get('/api/reports')

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['count'], 10)
        reports = data['reports']
        self.assertEqual(len(reports), 10)
        self.assertIn('supplier_id', reports[0])
        self.assertIn('risk_reason', reports[0])
        self.assertIn('recommended_action', reports[0])
        self.assertNotIn('report_name', reports[0])
        self.assertNotIn('date', reports[0])
        self.assertNotIn('status', reports[0])
        self.assertEqual(
            [report['overall_risk_score'] for report in reports],
            sorted((report['overall_risk_score'] for report in reports), reverse=True),
        )

    def test_reports_page_renders_matching_columns_and_data_loader(self):
        response = self.client.get('/reports')

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("fetch('/api/reports')", html)
        self.assertIn('Highest-Risk Supplier Summaries', html)
        self.assertIn('Recommended Action', html)
        self.assertIn('individual generated reports are not stored', html)
        self.assertNotIn('<th>Status</th>', html)
        self.assertNotIn('report_name}', html)
        self.assertNotIn('innerHTML', html)

    def test_reports_script_has_valid_javascript_syntax(self):
        import re
        import subprocess

        html = self.client.get('/reports').get_data(as_text=True)
        match = re.search(r'<script>(.*?)</script>', html, re.DOTALL)

        self.assertIsNotNone(match)
        result = subprocess.run(
            [
                'node',
                '-e',
                "let source=''; process.stdin.on('data', chunk => source += chunk); "
                "process.stdin.on('end', () => { new Function(source); });",
            ],
            input=match.group(1),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class SuppliersPageTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        with self.client.session_transaction() as session:
            session['user'] = 'suppliers-test'

    def test_suppliers_page_renders_search_filter_and_pagination(self):
        response = self.client.get('/suppliers')

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('id="supplier-search"', html)
        self.assertIn('id="risk-filter"', html)
        self.assertIn('id="previous-page"', html)
        self.assertIn('id="next-page"', html)
        self.assertIn("fetch('/api/suppliers')", html)
        self.assertNotIn('risk.innerHTML', html)

    def test_supplier_list_script_has_valid_javascript_syntax(self):
        import re
        import subprocess

        response = self.client.get('/suppliers')
        html = response.get_data(as_text=True)
        match = re.search(r'<script>(.*?)</script>', html, re.DOTALL)

        self.assertIsNotNone(match)
        result = subprocess.run(
            [
                'node',
                '-e',
                "let source=''; process.stdin.on('data', chunk => source += chunk); "
                "process.stdin.on('end', () => { new Function(source); });",
            ],
            input=match.group(1),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class DocumentUploadTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        with self.client.session_transaction() as session:
            session['user'] = 'upload-test'

    def test_upload_page_shows_validation_and_success_messages(self):
        with self.client:
            page = self.client.get('/upload')
            self.assertEqual(page.status_code, 200)
            self.assertIn('required', page.get_data(as_text=True))

            error_page = self.client.get('/upload?error=Upload+failed')
            self.assertIn('role="alert"', error_page.get_data(as_text=True))

            success_page = self.client.get('/upload?message=Upload+complete')
            self.assertIn('role="status"', success_page.get_data(as_text=True))

    def test_upload_requires_a_file_and_rejects_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as upload_dir, patch.dict(
            app.config, {'UPLOAD_FOLDER': upload_dir}
        ):
            missing = self.client.post('/upload-document', data={'supplier_name': 'Vendor'})
            unsupported = self.client.post(
                '/upload-document',
                data={'document': (BytesIO(b'not a supported document'), 'report.csv')},
                content_type='multipart/form-data'
            )

            self.assertEqual(missing.status_code, 400)
            self.assertIn('Choose a PDF or TXT', missing.get_data(as_text=True))
            self.assertEqual(unsupported.status_code, 400)
            self.assertIn('Please choose a PDF or TXT', unsupported.get_data(as_text=True))
            self.assertEqual(os.listdir(upload_dir), [])

    def test_upload_saves_with_safe_unique_name_and_confirms_processing(self):
        with tempfile.TemporaryDirectory() as upload_dir, patch.dict(
            app.config, {'UPLOAD_FOLDER': upload_dir}
        ), patch('app.ingest_document') as ingest:
            response = self.client.post(
                '/upload-document',
                data={
                    'supplier_name': 'Example Supplier',
                    'document': (BytesIO(b'Supplier risk review'), r'..\..\supplier.TXT')
                },
                content_type='multipart/form-data'
            )

            self.assertEqual(response.status_code, 302)
            saved_files = os.listdir(upload_dir)
            self.assertEqual(len(saved_files), 1)
            self.assertRegex(saved_files[0], re.compile(r'^[a-f0-9]{32}\.txt$'))
            self.assertEqual(os.path.dirname(ingest.call_args.args[0]), upload_dir)
            self.assertEqual(ingest.call_args.args[1], 'Example Supplier')

            confirmation = self.client.get(response.location)
            self.assertIn(
                'Document uploaded and processed successfully.',
                confirmation.get_data(as_text=True)
            )

    def test_document_without_extractable_text_is_rejected_and_removed(self):
        with tempfile.TemporaryDirectory() as upload_dir, patch.dict(
            app.config, {'UPLOAD_FOLDER': upload_dir}
        ):
            response = self.client.post(
                '/upload-document',
                data={'document': (BytesIO(b'   \n'), 'empty.txt')},
                content_type='multipart/form-data'
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn('No readable text', response.get_data(as_text=True))
            self.assertEqual(os.listdir(upload_dir), [])

    def test_assistant_upload_rejects_unsupported_extension_before_ai_request(self):
        with tempfile.TemporaryDirectory() as upload_dir, patch.dict(
            app.config, {'UPLOAD_FOLDER': upload_dir}
        ), patch('app.generate_ollama_response') as generate:
            response = self.client.post(
                '/analyze',
                data={
                    'query': 'Assess supplier risk.',
                    'supplier_id': 'SUP0004',
                    'document': (BytesIO(b'not an allowed document'), 'report.csv')
                },
                content_type='multipart/form-data'
            )

            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.get_json()['error'], 'Please choose a PDF or TXT document.')
            self.assertEqual(os.listdir(upload_dir), [])
            generate.assert_not_called()


class SettingsTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.temp_directory = tempfile.TemporaryDirectory()
        self.users_path = os.path.join(self.temp_directory.name, 'users.json')
        self.vector_store_path = os.path.join(self.temp_directory.name, 'vector_store.json')
        with open(self.users_path, 'w', encoding='utf-8') as users_file:
            json.dump({'admin': 'legacy-admin-password'}, users_file)
        with open(self.vector_store_path, 'w', encoding='utf-8') as store_file:
            json.dump([], store_file)
        self.config_patch = patch.dict(
            app.config,
            {'USERS_FILE': self.users_path, 'VECTOR_STORE_FILE': self.vector_store_path}
        )
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.addCleanup(self.temp_directory.cleanup)
        self.client = app.test_client()
        with self.client.session_transaction() as session:
            session['user'] = 'admin'

    def read_users(self):
        with open(self.users_path, encoding='utf-8') as users_file:
            return json.load(users_file)

    def test_settings_page_shows_provider_and_empty_vector_state(self):
        response = self.client.get('/settings')

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn(app.config['OLLAMA_MODEL'], html)
        self.assertIn('Email or alert notifications are not configured', html)
        self.assertIn('No documents have been indexed yet.', html)

    def test_new_user_password_is_hashed_and_can_authenticate(self):
        response = self.client.post(
            '/settings',
            data={'new_username': 'new-user', 'new_password': 'a secure password'}
        )

        self.assertEqual(response.status_code, 200)
        stored = self.read_users()['new-user']
        self.assertNotEqual(stored, 'a secure password')
        self.assertTrue(verify_user_password(stored, 'a secure password')[0])
        with self.client.session_transaction() as session:
            session.clear()
        login_response = self.client.post(
            '/login',
            data={'username': 'new-user', 'password': 'a secure password'}
        )
        self.assertEqual(login_response.status_code, 302)
        with self.client.session_transaction() as session:
            self.assertEqual(session.get('user'), 'new-user')

    def test_successful_legacy_plaintext_login_migrates_password_to_hash(self):
        response = self.client.post(
            '/login',
            data={'username': 'admin', 'password': 'legacy-admin-password'}
        )

        self.assertEqual(response.status_code, 302)
        stored = self.read_users()['admin']
        self.assertNotEqual(stored, 'legacy-admin-password')
        self.assertTrue(verify_user_password(stored, 'legacy-admin-password')[0])

    def test_duplicate_and_empty_user_submission_are_reported_without_overwrite(self):
        duplicate = self.client.post(
            '/settings',
            data={'new_username': 'admin', 'new_password': 'replacement'}
        )
        self.assertIn('That username already exists.', duplicate.get_data(as_text=True))
        self.assertEqual(self.read_users()['admin'], 'legacy-admin-password')

        empty = self.client.post('/settings', data={'new_username': '', 'new_password': ''})
        self.assertIn('Enter both a username and password.', empty.get_data(as_text=True))

    def test_admin_cannot_delete_admin_account(self):
        response = self.client.post('/settings', data={'delete_username': 'admin'})

        self.assertIn('Cannot remove admin or unknown user.', response.get_data(as_text=True))
        self.assertIn('admin', self.read_users())

    def test_non_admin_cannot_manage_users_or_ingest_documents(self):
        with self.client.session_transaction() as session:
            session['user'] = 'regular-user'

        response = self.client.post(
            '/settings',
            data={'new_username': 'unauthorized', 'new_password': 'secret'}
        )
        self.assertIn('Only admin can manage users.', response.get_data(as_text=True))
        self.assertNotIn('unauthorized', self.read_users())

        with patch('app.ingest_existing_uploads') as ingest:
            denied = self.client.post('/ingest-existing-uploads', follow_redirects=True)
        ingest.assert_not_called()
        self.assertEqual(denied.status_code, 200)
        self.assertIn('Only admin can ingest documents.', denied.get_data(as_text=True))


class ProductionUserSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.users_path = os.path.join(self.temp_directory.name, 'users.json')
        self.config_patch = patch.dict(app.config, {'USERS_FILE': self.users_path})
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.addCleanup(self.temp_directory.cleanup)

    def test_first_production_login_creates_hashed_admin_from_environment(self):
        with patch('app.PRODUCTION', True), patch.dict(os.environ, {
            'INITIAL_ADMIN_USERNAME': 'deployer',
            'INITIAL_ADMIN_PASSWORD': 'a secure password',
        }):
            users = load_users()

        self.assertTrue(verify_user_password(users['deployer'], 'a secure password')[0])
        self.assertNotEqual(users['deployer'], 'a secure password')

    def test_production_does_not_create_default_users_without_setup_credentials(self):
        with patch('app.PRODUCTION', True), patch.dict(os.environ, {
            'INITIAL_ADMIN_USERNAME': '',
            'INITIAL_ADMIN_PASSWORD': '',
        }):
            with self.assertRaisesRegex(RuntimeError, 'INITIAL_ADMIN_USERNAME'):
                load_users()

        self.assertFalse(os.path.exists(self.users_path))


if __name__ == '__main__':
    unittest.main()

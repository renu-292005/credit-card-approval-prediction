import unittest
from app import app

class CreditCardAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_route(self):
        """Test that home landing page loads correctly."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Credit Shield AI', response.data)
        self.assertIn(b'Automated Underwriting', response.data)

    def test_predict_form_route(self):
        """Test that form page loads correctly."""
        response = self.app.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Applicant Profile', response.data)

    def test_prediction_approved_case(self):
        """Test prediction endpoint with a profile that should be approved."""
        # Strong profile: high income, zero debt, high credit score
        payload = {
            'age': '35',
            'income': '120000',
            'debt': '0',
            'credit_score': '820'
        }
        response = self.app.post('/get_prediction', data=payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Approved', response.data)

    def test_prediction_rejected_case(self):
        """Test prediction endpoint with a profile that should be rejected."""
        # Weak profile: low income, high debt, low credit score
        payload = {
            'age': '22',
            'income': '15000',
            'debt': '20000',
            'credit_score': '350'
        }
        response = self.app.post('/get_prediction', data=payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Declined', response.data)

    def test_validation_invalid_fico(self):
        """Test that credit score boundary checks reject invalid values."""
        # Out-of-bounds credit score (900)
        payload = {
            'age': '30',
            'income': '50000',
            'debt': '1000',
            'credit_score': '900'
        }
        response = self.app.post('/get_prediction', data=payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Credit Score must be between 300 and 850.', response.data)

if __name__ == '__main__':
    unittest.main()

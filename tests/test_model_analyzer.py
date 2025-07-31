import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ai_bias_detector import ModelAnalyzer

@pytest.fixture
def sample_data_for_model():
    """Menyediakan data sampel dan model terlatih untuk pengujian."""
    data = {
        'fitur1': [i for i in range(20)],
        'fitur2': [i % 4 for i in range(20)],
        'gender': ['Pria'] * 10 + ['Wanita'] * 10,
        'target': [1, 0] * 10
    }
    df = pd.DataFrame(data)

    X = df[['fitur1', 'fitur2']]
    y = df['target']

    model = LogisticRegression()
    model.fit(X, y)

    return model, df

def test_model_analyzer_init(sample_data_for_model):
    """Menguji inisialisasi ModelAnalyzer."""
    model, df = sample_data_for_model
    analyzer = ModelAnalyzer(model, df, target_variable='target', sensitive_attributes=['gender'])
    assert analyzer.model == model
    assert analyzer.df.equals(df)
    assert analyzer.target_variable == 'target'
    assert analyzer.sensitive_attributes == ['gender']

def test_analyze_structure(sample_data_for_model):
    """Menguji apakah metode analyze() menghasilkan struktur output yang benar."""
    model, df = sample_data_for_model
    analyzer = ModelAnalyzer(model, df, target_variable='target', sensitive_attributes=['gender'])
    results = analyzer.analyze()

    assert 'overall' in results
    assert 'gender' in results
    assert 'Pria' in results['gender']
    assert 'Wanita' in results['gender']

    # Periksa apakah semua metrik ada untuk satu kelompok
    assert 'accuracy' in results['gender']['Pria']
    assert 'precision' in results['gender']['Pria']
    assert 'recall' in results['gender']['Pria']

def test_generate_report(sample_data_for_model):
    """Menguji pembuatan laporan."""
    model, df = sample_data_for_model
    analyzer = ModelAnalyzer(model, df, target_variable='target', sensitive_attributes=['gender'])
    report = analyzer.generate_report()

    assert "Laporan Analisis Bias Model" in report
    assert "Kinerja Keseluruhan" in report
    assert "Analisis Kinerja berdasarkan 'gender'" in report
    assert "Kelompok 'Pria'" in report
    assert "Kelompok 'Wanita'" in report
    assert "Akurasi" in report
    assert "Presisi" in report
    assert "Recall" in report

import pytest
import pandas as pd
from ai_bias_detector import DatasetAnalyzer

@pytest.fixture
def sample_biased_df():
    """Menyediakan DataFrame sampel dengan bias yang diketahui untuk pengujian."""
    data = {
        'gender': ['Pria', 'Wanita', 'Pria', 'Pria', 'Wanita'],
        'ras': ['A', 'B', 'A', 'A', 'C']
    }
    return pd.DataFrame(data)

def test_dataset_analyzer_init(sample_biased_df):
    """Menguji inisialisasi DatasetAnalyzer."""
    analyzer = DatasetAnalyzer(sample_biased_df, sensitive_attributes=['gender', 'ras'])
    assert analyzer.df.equals(sample_biased_df)
    assert analyzer.sensitive_attributes == ['gender', 'ras']

def test_dataset_analyzer_init_raises_error_on_missing_attribute(sample_biased_df):
    """Menguji bahwa ValueError muncul jika atribut sensitif tidak ada."""
    with pytest.raises(ValueError, match="Atribut sensitif berikut tidak ditemukan"):
        DatasetAnalyzer(sample_biased_df, sensitive_attributes=['usia'])

def test_analyze_distribution_and_counts(sample_biased_df):
    """Menguji apakah metode analyze() menghitung distribusi dan jumlah dengan benar."""
    analyzer = DatasetAnalyzer(sample_biased_df, sensitive_attributes=['gender'])
    results = analyzer.analyze()

    # Memeriksa analisis gender
    gender_analysis = results.get('gender', {})
    assert gender_analysis is not None

    # Memeriksa distribusi (proporsi)
    assert gender_analysis['distribution']['Pria'] == pytest.approx(0.6)
    assert gender_analysis['distribution']['Wanita'] == pytest.approx(0.4)

    # Memeriksa jumlah
    assert gender_analysis['counts']['Pria'] == 3
    assert gender_analysis['counts']['Wanita'] == 2

def test_generate_report(sample_biased_df):
    """Menguji pembuatan laporan."""
    analyzer = DatasetAnalyzer(sample_biased_df, sensitive_attributes=['gender'])
    report = analyzer.generate_report()

    assert "Laporan Analisis Bias Dataset" in report
    assert "Analisis untuk atribut: 'gender'" in report
    assert "Kategori 'Pria': 60.00%" in report
    assert "Kategori 'Wanita': 40.00%" in report

def test_generate_detailed_report(sample_biased_df):
    """Menguji pembuatan laporan terperinci."""
    analyzer = DatasetAnalyzer(sample_biased_df, sensitive_attributes=['ras'])
    report = analyzer.generate_report(detailed=True)

    assert "Kategori 'A': 60.00% (3 sampel)" in report
    assert "Kategori 'B': 20.00% (1 sampel)" in report
    assert "Kategori 'C': 20.00% (1 sampel)" in report

import pandas as pd
from typing import List, Dict, Any

class DatasetAnalyzer:
    """
    Menganalisis bias dalam dataset dengan memeriksa distribusi atribut sensitif.
    """

    def __init__(self, df: pd.DataFrame, sensitive_attributes: List[str]):
        """
        Menginisialisasi DatasetAnalyzer.

        Args:
            df (pd.DataFrame): DataFrame pandas yang akan dianalisis.
            sensitive_attributes (List[str]): Daftar nama kolom yang berisi atribut sensitif (misalnya, ['gender', 'ras']).
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' harus berupa instance pandas DataFrame.")

        if not sensitive_attributes:
            raise ValueError("Daftar 'sensitive_attributes' tidak boleh kosong.")

        missing_attrs = [attr for attr in sensitive_attributes if attr not in df.columns]
        if missing_attrs:
            raise ValueError(f"Atribut sensitif berikut tidak ditemukan di DataFrame: {', '.join(missing_attrs)}")

        self.df = df
        self.sensitive_attributes = sensitive_attributes
        self._analysis_results: Dict[str, Any] = {}

    def analyze(self) -> Dict[str, Any]:
        """
        Melakukan analisis bias dengan menghitung distribusi setiap atribut sensitif.

        Returns:
            Dict[str, Any]: Kamus yang berisi hasil analisis.
        """
        self._analysis_results = {}
        for attr in self.sensitive_attributes:
            distribution = self.df[attr].value_counts(normalize=True)
            counts = self.df[attr].value_counts(normalize=False)

            self._analysis_results[attr] = {
                'distribution': distribution.to_dict(),
                'counts': counts.to_dict()
            }
        return self._analysis_results

    def generate_report(self, detailed: bool = False) -> str:
        """
        Menghasilkan laporan tekstual yang merangkum bias dataset.

        Args:
            detailed (bool): Jika True, sertakan jumlah mentah dalam laporan.

        Returns:
            str: String laporan yang diformat.
        """
        if not self._analysis_results:
            self.analyze()

        report_lines = ["Laporan Analisis Bias Dataset", "="*30]

        for attr, results in self._analysis_results.items():
            report_lines.append(f"\nAnalisis untuk atribut: '{attr}'")
            report_lines.append("-" * (25 + len(attr)))

            if not results['distribution']:
                report_lines.append("Tidak ada data untuk dianalisis.")
                continue

            for category, percentage in results['distribution'].items():
                count = results['counts'][category]
                if detailed:
                    report_lines.append(f"- Kategori '{category}': {percentage:.2%} ({count} sampel)")
                else:
                    report_lines.append(f"- Kategori '{category}': {percentage:.2%}")

        return "\n".join(report_lines)

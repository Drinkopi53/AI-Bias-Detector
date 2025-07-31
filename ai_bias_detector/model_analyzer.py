import pandas as pd
from typing import List, Dict, Any
from sklearn.base import is_classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ModelAnalyzer:
    """
    Menganalisis bias dalam model machine learning dengan mengevaluasi kinerjanya
    di berbagai kelompok demografis.
    """

    def __init__(self, model: Any, df: pd.DataFrame, target_variable: str, sensitive_attributes: List[str]):
        """
        Menginisialisasi ModelAnalyzer.

        Args:
            model (Any): Model klasifikasi terlatih yang kompatibel dengan Scikit-learn.
            df (pd.DataFrame): DataFrame pengujian yang berisi fitur, variabel target, dan atribut sensitif.
            target_variable (str): Nama kolom variabel target.
            sensitive_attributes (List[str]): Daftar nama kolom yang berisi atribut sensitif.
        """
        if not hasattr(model, 'predict'):
            raise TypeError("Model harus memiliki metode 'predict'.")
        if not is_classifier(model):
            raise TypeError("Model harus berupa classifier yang kompatibel dengan Scikit-learn.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' harus berupa instance pandas DataFrame.")
        if target_variable not in df.columns:
            raise ValueError(f"Variabel target '{target_variable}' tidak ditemukan di DataFrame.")

        self.model = model
        self.df = df
        self.target_variable = target_variable
        self.sensitive_attributes = sensitive_attributes
        self.features = [col for col in df.columns if col not in [target_variable] + sensitive_attributes]
        self._analysis_results: Dict[str, Any] = {}

    def analyze(self) -> Dict[str, Any]:
        """
        Mengevaluasi kinerja model di berbagai kelompok demografis.

        Returns:
            Dict[str, Any]: Kamus yang berisi metrik kinerja untuk setiap subkelompok.
        """
        X = self.df[self.features]
        y_true = self.df[self.target_variable]

        # Analisis keseluruhan
        y_pred_overall = self.model.predict(X)
        self._analysis_results['overall'] = self._calculate_metrics(y_true, y_pred_overall)

        # Analisis per subkelompok
        for attr in self.sensitive_attributes:
            self._analysis_results[attr] = {}
            for group in self.df[attr].unique():
                group_df = self.df[self.df[attr] == group]
                if group_df.empty:
                    continue

                X_group = group_df[self.features]
                y_true_group = group_df[self.target_variable]
                y_pred_group = self.model.predict(X_group)

                self._analysis_results[attr][group] = self._calculate_metrics(y_true_group, y_pred_group)

        return self._analysis_results

    def _calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Menghitung metrik kinerja."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def generate_report(self) -> str:
        """
        Menghasilkan laporan tekstual yang merangkum disparitas kinerja model.

        Returns:
            str: String laporan yang diformat.
        """
        if not self._analysis_results:
            self.analyze()

        report_lines = ["Laporan Analisis Bias Model", "="*30]
        overall_metrics = self._analysis_results.get('overall', {})
        report_lines.append(f"\nKinerja Keseluruhan:\n  - Akurasi: {overall_metrics.get('accuracy', 0):.2%}\n  - Presisi: {overall_metrics.get('precision', 0):.2%}\n  - Recall: {overall_metrics.get('recall', 0):.2%}")

        for attr, groups in self._analysis_results.items():
            if attr == 'overall':
                continue
            report_lines.append(f"\nAnalisis Kinerja berdasarkan '{attr}':")
            report_lines.append("-" * (30 + len(attr)))
            for group, metrics in groups.items():
                report_lines.append(f"  Kelompok '{group}':")
                report_lines.append(f"    - Akurasi: {metrics['accuracy']:.2%}")
                report_lines.append(f"    - Presisi: {metrics['precision']:.2%}")
                report_lines.append(f"    - Recall: {metrics['recall']:.2%}")

        return "\n".join(report_lines)

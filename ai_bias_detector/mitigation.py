from typing import List, Dict, Any

class MitigationRecommender:
    """
    Memberikan rekomendasi untuk memitigasi bias yang terdeteksi dalam dataset dan model.
    """

    def __init__(self, dataset_analysis: Dict[str, Any] = None, model_analysis: Dict[str, Any] = None):
        """
        Menginisialisasi MitigationRecommender.

        Args:
            dataset_analysis (Dict[str, Any], opsional): Hasil dari DatasetAnalyzer.analyze().
            model_analysis (Dict[str, Any], opsional): Hasil dari ModelAnalyzer.analyze().
        """
        self.dataset_analysis = dataset_analysis if dataset_analysis else {}
        self.model_analysis = model_analysis if model_analysis else {}

    def generate_recommendations(self, imbalance_threshold: float = 0.5, performance_gap_threshold: float = 0.1) -> List[str]:
        """
        Menghasilkan daftar rekomendasi mitigasi bias yang dapat ditindaklanjuti.

        Args:
            imbalance_threshold (float): Ambang batas untuk rasio distribusi kelompok minoritas terhadap mayoritas
                                         untuk memicu rekomendasi penyeimbangan ulang data.
            performance_gap_threshold (float): Ambang batas perbedaan metrik kinerja antar kelompok
                                               untuk memicu rekomendasi keadilan algoritmik.

        Returns:
            List[str]: Daftar string rekomendasi.
        """
        recommendations = []

        # Rekomendasi berdasarkan analisis dataset
        if self.dataset_analysis:
            for attr, results in self.dataset_analysis.items():
                dist = results.get('distribution', {})
                if len(dist) > 1:
                    max_dist = max(dist.values())
                    min_dist = min(dist.values())
                    if min_dist / max_dist < imbalance_threshold:
                        minority_group = min(dist, key=dist.get)
                        majority_group = max(dist, key=dist.get)
                        recommendations.append(
                            f"Dataset: Atribut '{attr}' menunjukkan ketidakseimbangan yang signifikan. "
                            f"Pertimbangkan untuk menggunakan teknik penyeimbangan ulang seperti oversampling pada '{minority_group}' "
                            f"atau undersampling pada '{majority_group}'."
                        )

        # Rekomendasi berdasarkan analisis model
        if self.model_analysis:
            for attr, groups in self.model_analysis.items():
                if attr == 'overall' or not isinstance(groups, dict):
                    continue

                # Kumpulkan metrik untuk semua kelompok dalam satu atribut
                metrics = {metric: [data[metric] for data in groups.values()] for metric in ['accuracy', 'precision', 'recall']}

                for metric_name, values in metrics.items():
                    if len(values) > 1:
                        max_perf = max(values)
                        min_perf = min(values)
                        if max_perf - min_perf > performance_gap_threshold:
                            worst_group = list(groups.keys())[values.index(min_perf)]
                            best_group = list(groups.keys())[values.index(max_perf)]
                            recommendations.append(
                                f"Model: Terdapat kesenjangan kinerja {metric_name} yang signifikan (> {performance_gap_threshold:.0%}) "
                                f"pada atribut '{attr}' (antara {worst_group} dan {best_group}). "
                                "Pertimbangkan teknik keadilan algoritmik (misalnya, re-weighting, post-processing)."
                            )

        if not recommendations:
            recommendations.append("Tidak ada masalah bias yang signifikan yang terdeteksi berdasarkan ambang batas saat ini.")

        return recommendations

"""
agent.py
========

Kerangka kerja untuk semua 'otak' atau 'agent' pembuat keputusan.
Setiap agent harus mewarisi dari BaseAgent dan mengimplementasikan
metode `decide()`.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

# Konfigurasi logging dasar
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BaseAgent(ABC):
    """
    Kelas dasar abstrak untuk semua agent.
    """
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    def decide(self, opportunity: Dict[str, Any]) -> str:
        """
        Metode utama untuk membuat keputusan trading.
        
        Args:
            opportunity (Dict[str, Any]): Sebuah dictionary yang berisi semua
                                         informasi tentang peluang trading
                                         (skor, komponen, fitur, dll.).
        
        Returns:
            str: Keputusan trading, contoh: "ACCEPT", "REJECT", "WAIT".
        """
        pass

    def __str__(self):
        return f"Agent(name={self.name})"


class RuleBasedAgent(BaseAgent):
    """
    Agent sederhana yang membuat keputusan berdasarkan aturan-aturan dasar.
    Contoh: Cek skor dan beberapa fitur kunci.
    """
    def __init__(self, confidence_threshold: float = 5.0):
        super().__init__(name="RuleBasedAgent")
        self.confidence_threshold = confidence_threshold
        self.logger.info(f"Agent diinisialisasi dengan threshold: {self.confidence_threshold}")

    def decide(self, opportunity: Dict[str, Any]) -> str:
        """
        Membuat keputusan berdasarkan skor dari sinyal.
        """
        score = opportunity.get('score', 0)

        if abs(score) >= self.confidence_threshold:
            decision = "ACCEPT"
            self.logger.info(f"Keputusan: {decision} (Skor {score:.2f} >= Threshold {self.confidence_threshold})")
        else:
            decision = "REJECT"
            self.logger.info(f"Keputusan: {decision} (Skor {score:.2f} < Threshold {self.confidence_threshold})")
            
        return decision


# --- Agent Cerdas dengan Model Machine Learning ---
import joblib
import numpy as np
from sklearn.base import BaseEstimator

class NeuralAgent(BaseAgent):
    """
    Agent yang menggunakan model Jaringan Syaraf Tiruan (atau model scikit-learn lainnya)
    yang sudah dilatih untuk membuat keputusan.
    """
    def __init__(self, model_path: str, probability_threshold: float = 0.70):
        super().__init__(name="NeuralAgent")
        self.model = self._load_model(model_path)
        self.probability_threshold = probability_threshold
        self.logger.info(f"Agent diinisialisasi dengan model dari: {model_path} dan threshold probabilitas: {self.probability_threshold}")

    def _load_model(self, model_path: str) -> Optional[BaseEstimator]:
        """Memuat model yang sudah dilatih dari file."""
        try:
            model = joblib.load(model_path)
            self.logger.info(f"Model {type(model).__name__} berhasil dimuat.")
            return model
        except FileNotFoundError:
            self.logger.error(f"File model tidak ditemukan di {model_path}. Agent tidak akan berfungsi.")
            return None
        except Exception as e:
            self.logger.error(f"Gagal memuat model dari {model_path}: {e}")
            return None

    def decide(self, opportunity: Dict[str, Any]) -> str:
        """
        Membuat keputusan menggunakan prediksi dari model neural network.
        """
        if self.model is None:
            self.logger.warning("Model tidak tersedia, keputusan otomatis REJECT.")
            return "REJECT"

        features = opportunity.get('features')
        if features is None or not hasattr(features, 'size') or features.size == 0:
            self.logger.warning("Tidak ada fitur (features) untuk dianalisis, keputusan REJECT.")
            return "REJECT"

        # Model scikit-learn mengharapkan input 2D
        features_2d = np.array(features).reshape(1, -1)
        
        try:
            # Memeriksa apakah model memiliki metode predict_proba
            if not hasattr(self.model, 'predict_proba'):
                self.logger.warning(f"Model {type(self.model).__name__} tidak memiliki 'predict_proba'. Menggunakan 'predict'.")
                prediction = self.model.predict(features_2d)[0]
                # Asumsikan '1' atau 'ACCEPT' adalah sinyal positif
                decision = "ACCEPT" if str(prediction).upper() in ["1", "ACCEPT"] else "REJECT"
            else:
                # Prediksi probabilitas (asumsi kelas 1 adalah 'WIN' atau 'ACCEPT')
                probabilities = self.model.predict_proba(features_2d)[0]
                if len(probabilities) < 2:
                    self.logger.error(f"Model hanya mengembalikan {len(probabilities)} probabilitas. Butuh setidaknya 2.")
                    return "REJECT"

                probability_of_win = probabilities[1]
                
                if probability_of_win > self.probability_threshold:
                    decision = "ACCEPT"
                else:
                    decision = "REJECT"
                self.logger.info(f"Probabilitas keberhasilan: {probability_of_win:.2%}. Keputusan: {decision}")

            return decision

        except Exception as e:
            self.logger.error(f"Gagal membuat prediksi: {e}", exc_info=True)
            return "REJECT"

class EnsembleAgent(BaseAgent):
    """
    Agent yang menggabungkan keputusan dari beberapa agent lain untuk
    mencapai konsensus.
    """
    def __init__(self, agents: List[BaseAgent], strategy: str = 'majority_vote'):
        super().__init__(name="EnsembleAgent")
        if not agents:
            raise ValueError("EnsembleAgent memerlukan setidaknya satu agent.")
        self.agents = agents
        self.strategy = strategy
        self.logger.info(f"Agent diinisialisasi dengan {len(self.agents)} sub-agents dan strategi: {self.strategy}")

    def decide(self, opportunity: Dict[str, Any]) -> str:
        """
        Menjalankan strategi voting untuk membuat keputusan akhir.
        """
        decisions = []
        for agent in self.agents:
            try:
                decision = agent.decide(opportunity)
                decisions.append(decision)
                self.logger.debug(f"Agent {agent.name} memutuskan: {decision}")
            except Exception as e:
                self.logger.error(f"Agent {agent.name} gagal membuat keputusan: {e}")
                decisions.append("REJECT") # Anggap REJECT jika ada error

        if not decisions:
            self.logger.warning("Tidak ada keputusan dari sub-agents, mengembalikan REJECT.")
            return "REJECT"

        if self.strategy == 'majority_vote':
            return self._majority_vote(decisions)
        else:
            self.logger.error(f"Strategi '{self.strategy}' tidak diketahui. Menggunakan REJECT.")
            return "REJECT"

    def _majority_vote(self, decisions: List[str]) -> str:
        """Menentukan keputusan berdasarkan suara mayoritas."""
        votes = {"ACCEPT": 0, "REJECT": 0, "WAIT": 0}
        for d in decisions:
            if d in votes:
                votes[d] += 1
        
        # Log detail voting
        self.logger.info(f"Hasil voting: {votes}")

        # Prioritaskan ACCEPT jika imbang antara ACCEPT dan REJECT
        if votes["ACCEPT"] > 0 and votes["ACCEPT"] >= votes["REJECT"]:
            final_decision = "ACCEPT"
        elif votes["REJECT"] > votes["ACCEPT"]:
            final_decision = "REJECT"
        else: # Jika hanya ada WAIT atau tidak ada mayoritas jelas
            final_decision = "WAIT"
            
        self.logger.info(f"Keputusan akhir (mayoritas): {final_decision}")
        return final_decision

def create_agent(config: Dict[str, Any]) -> BaseAgent:
    """
    Factory function untuk membuat instance agent berdasarkan konfigurasi.
    """
    agent_type = config.get("type")
    params = config.get("params", {})
    
    if agent_type == "rule_based":
        return RuleBasedAgent(**params)
    elif agent_type == "neural":
        return NeuralAgent(**params)
    elif agent_type == "ensemble":
        sub_agents_configs = params.get("agents", [])
        sub_agents = [create_agent(conf) for conf in sub_agents_configs]
        strategy = params.get("strategy", "majority_vote")
        return EnsembleAgent(agents=sub_agents, strategy=strategy)
    else:
        raise ValueError(f"Tipe agent tidak diketahui: {agent_type}")

# Contoh Penggunaan:
if __name__ == '__main__':
    # Contoh ini hanya akan berjalan jika file dieksekusi secara langsung
    # dan memerlukan file model dummy.
    
    # 1. Buat file model dummy untuk pengujian
    from sklearn.linear_model import LogisticRegression
    import os
    
    dummy_model_path = "dummy_model.joblib"
    if not os.path.exists(dummy_model_path):
        # Latih model sederhana
        X_train = np.random.rand(10, 5)
        y_train = (np.sum(X_train, axis=1) > 2.5).astype(int)
        dummy_model = LogisticRegression()
        dummy_model.fit(X_train, y_train)
        joblib.dump(dummy_model, dummy_model_path)
        print(f"Model dummy '{dummy_model_path}' dibuat.")

    # 2. Konfigurasi untuk membuat agent
    agent_config = {
        "type": "ensemble",
        "params": {
            "strategy": "majority_vote",
            "agents": [
                {
                    "type": "rule_based",
                    "params": {"confidence_threshold": 6.0}
                },
                {
                    "type": "neural",
                    "params": {
                        "model_path": dummy_model_path,
                        "probability_threshold": 0.65
                    }
                }
            ]
        }
    }

    # 3. Buat agent utama dari konfigurasi
    main_agent = create_agent(agent_config)
    
    # 4. Siapkan data peluang dummy
    dummy_opportunity = {
        "score": 7.5,
        "features": np.random.rand(5) # Fitur harus sesuai dengan yang diharapkan model
    }
    
    # 5. Dapatkan keputusan dari agent
    print("-" * 30)
    final_decision = main_agent.decide(dummy_opportunity)
    print(f"\nKEPUTUSAN FINAL DARI {main_agent.name}: {final_decision}")
    print("-" * 30)

    # 6. Hapus file dummy
    os.remove(dummy_model_path)
    print(f"Model dummy '{dummy_model_path}' dihapus.")
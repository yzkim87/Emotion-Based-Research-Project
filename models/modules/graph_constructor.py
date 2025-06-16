import torch

class GraphConstructor:
    """
    대화 utterance 간 관계를 나타내는 Graph adjacency matrix 생성기.
    기본: window size 또는 speaker ID 기반 연결.
    """

    def __init__(self, window_size=10):
        """
        Args:
            window_size (int): 문맥 연결 범위
        """
        self.window_size = window_size

    def build_adjacency(self, speaker_ids):
        """
        Args:
            speaker_ids (List[int]): 각 utterance의 speaker index
                예: [0,0,1,1,0,0,...]

        Returns:
            edge_index: [2, num_edges] (PyG 형식)
            edge_weight: [num_edges]
        """
        num_nodes = len(speaker_ids)
        edges = []

        for i in range(num_nodes):
            # window 안 노드와 연결
            for j in range(max(0, i - self.window_size), min(num_nodes, i + self.window_size + 1)):
                if i == j:
                    continue
                # 같은 화자면 연결 강화 예: weight 1.0 + alpha
                if speaker_ids[i] == speaker_ids[j]:
                    edges.append((i, j))
                else:
                    edges.append((i, j))

        # 중복 제거
        edges = list(set(edges))

        # edge_index [2, num_edges]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # 간단히 weight = 1.0 (화자 동일하면 향후 alpha 가중치 적용 가능)
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float)

        return edge_index, edge_weight

# test_pred_format.py
import pickle
import numpy as np
import torch


def analyze_pred_format():
    """Анализирует формат pred.predictions"""

    # Создаем тестовые данные для анализа
    test_cases = [
        ("torch_3d", torch.randn(4, 20, 50265)),  # Logits как torch tensor
        ("numpy_3d", np.random.randn(4, 20, 50265)),  # Logits как numpy
        ("torch_2d", torch.randint(0, 100, (4, 20))),  # Token IDs как torch
        ("numpy_2d", np.random.randint(0, 100, (4, 20))),  # Token IDs как numpy
        ("tuple", (torch.randn(4, 20, 50265),)),  # Кортеж с logits
    ]

    for name, test_data in test_cases:
        print(f"\n🔍 Тест: {name}")
        print(f"   Тип: {type(test_data)}")

        if hasattr(test_data, 'shape'):
            print(f"   Форма: {test_data.shape}")

        # Пробуем argmax
        if isinstance(test_data, torch.Tensor) and len(test_data.shape) == 3:
            try:
                result = test_data.argmax(dim=-1)
                print(f"   torch.argmax(dim=-1): OK, форма {result.shape}")
            except Exception as e:
                print(f"   torch.argmax(dim=-1): Ошибка - {e}")

        if isinstance(test_data, np.ndarray) and len(test_data.shape) == 3:
            try:
                result = test_data.argmax(axis=-1)
                print(f"   numpy.argmax(axis=-1): OK, форма {result.shape}")
            except Exception as e:
                print(f"   numpy.argmax(axis=-1): Ошибка - {e}")


# Запуск анализа
if __name__ == "__main__":
    analyze_pred_format()

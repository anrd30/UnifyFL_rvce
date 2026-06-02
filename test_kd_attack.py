import os
import torch
import torch.nn as nn
from models.cifar import CIFAR10Model

def test_kd_and_poisoning():
    print("=== Testing KD and Logit Poisoning Attack ===")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Student and Teacher models
    student = CIFAR10Model().to(device)
    teacher = CIFAR10Model().to(device)
    
    # Create fake dataloader with 2 batches of random images
    fake_images = torch.randn(64, 3, 32, 32)
    fake_labels = torch.randint(0, 10, (64,))
    
    class FakeLoader:
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
        def __len__(self):
            return 2
        def __iter__(self):
            for i in range(0, len(self.images), 32):
                yield {"image": self.images[i:i+32], "label": self.labels[i:i+32]}
                
    trainloader = FakeLoader(fake_images, fake_labels)
    
    # 2. Test standard training (No KD)
    print("\n--- Testing Standard Training (No KD) ---")
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    
    # Run 1 epoch
    os.environ["USE_KD"] = "false"
    os.environ["IS_MALICIOUS"] = "false"
    student.train_model(trainloader, epochs=1, optimizer=optimizer)
    print("✓ Standard training completed successfully")
    
    # 3. Test KD training (With Teacher)
    print("\n--- Testing KD Training (With Teacher) ---")
    os.environ["USE_KD"] = "true"
    os.environ["IS_MALICIOUS"] = "false"
    
    # Reset student model
    student = CIFAR10Model().to(device)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    
    student.train_model(trainloader, epochs=1, optimizer=optimizer, teacher_model=teacher, alpha=0.5, temperature=2.0)
    print("✓ KD training completed successfully")
    
    # 4. Test KD training with Logit Poisoning Attack
    print("\n--- Testing KD Training with Logit Poisoning Attack ---")
    os.environ["USE_KD"] = "true"
    os.environ["IS_MALICIOUS"] = "true"
    os.environ["ATTACK_TYPE"] = "logit_poisoning"
    os.environ["POISON_SCALE"] = "2.0"
    
    student = CIFAR10Model().to(device)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    
    # Run training and ensure it doesn't crash, and check that the weights get updated
    initial_weights = [p.clone().detach() for p in student.parameters()]
    student.train_model(trainloader, epochs=1, optimizer=optimizer, teacher_model=teacher, alpha=0.5, temperature=2.0)
    
    # Verify weights changed
    weights_changed = False
    for p_init, p_curr in zip(initial_weights, student.parameters()):
        if not torch.allclose(p_init, p_curr):
            weights_changed = True
            break
            
    assert weights_changed, "Student model weights did not update during logit poisoning!"
    print("✓ Logit poisoning training completed and weights were updated")
    
    # Clean up environment variables
    for var in ["USE_KD", "IS_MALICIOUS", "ATTACK_TYPE", "POISON_SCALE"]:
        if var in os.environ:
            del os.environ[var]
            
    print("\n=== All Tests Passed successfully! ===")

if __name__ == "__main__":
    test_kd_and_poisoning()

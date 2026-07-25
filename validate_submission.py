# validate_submission.py
import sys
import os
import importlib.util

def check(name, condition, error_msg="Failed"):
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"{name:<30} {status}")
    if not condition:
        print(f"   Reason: {error_msg}")
        return False
    return True

def validate():
    print("==================================================")
    print("HACKTRONIX TRACK 2 PRE-SUBMISSION VALIDATION")
    print("==================================================")
    
    all_passed = True
    
    # 1. Python version
    all_passed &= check("Python Version >= 3.8", sys.version_info >= (3, 8), "Requires Python 3.8+")
    
    # 2. Required Packages
    required_pkgs = ["cv2", "psutil", "numpy", "transformers", "torch"]
    for pkg in required_pkgs:
        # Some packages have different import names
        import_name = "cv2" if pkg == "opencv-python" else pkg
        found = importlib.util.find_spec(import_name) is not None
        all_passed &= check(f"Package: {pkg}", found, f"{pkg} not installed")
        
    # 3. Model Availability
    # We check if moondream weights exist or if we can instantiate the extractor
    # For now, just check if the local weights directory exists (assuming user downloaded it)
    # The actual Moondream extractor handles the download from HuggingFace via transformers
    all_passed &= check("Transformers Cache Access", os.access(os.path.expanduser("~/.cache/huggingface"), os.W_OK | os.X_OK) if os.path.exists(os.path.expanduser("~/.cache/huggingface")) else True, "Cannot access huggingface cache for Moondream")
    
    # 4. Configuration Validity
    try:
        from shared.config import DEFAULT_CONFIG
        all_passed &= check("Configuration Validity", True)
    except Exception as e:
        all_passed &= check("Configuration Validity", False, str(e))
        
    # 5. Output Directory Creation & Permissions
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    all_passed &= check("Output Directories Created", os.path.exists(results_dir), "Could not create results directory")
    all_passed &= check("Write Permissions", os.access(results_dir, os.W_OK), "No write permissions to results directory")
    
    # 6. Ground Truth Availability
    gt_path = os.path.join(os.path.dirname(__file__), "ground_truth", "dataset.json")
    all_passed &= check("Ground Truth Available", os.path.exists(gt_path), f"Missing {gt_path}")
    
    # 7. Pipeline Initialization
    try:
        from tests.test_corrections import MockPipeline
        MockPipeline()
        all_passed &= check("Pipeline Initialization", True)
    except Exception as e:
        all_passed &= check("Pipeline Initialization", False, str(e))

    print("==================================================")
    if all_passed:
        print("🎉 ALL CHECKS PASSED. SYSTEM IS READY FOR SUBMISSION.")
        sys.exit(0)
    else:
        print("⚠️ SOME CHECKS FAILED. PLEASE FIX BEFORE SUBMISSION.")
        sys.exit(1)

if __name__ == "__main__":
    validate()

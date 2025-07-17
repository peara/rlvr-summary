#!/usr/bin/env python3
"""Test script to verify FENICE integration without requiring all dependencies."""

import sys
import os
sys.path.insert(0, 'src')

def test_import_structure():
    """Test that the import structure is correct."""
    print("Testing import structure...")
    
    # Test that the vendor package exists
    try:
        import rlvr_summary.vendor
        print("✅ Vendor package imports successfully")
    except ImportError as e:
        print(f"❌ Vendor package import failed: {e}")
        return False
    
    # Test that the package structure is correct
    try:
        vendor_path = os.path.join('src', 'rlvr_summary', 'vendor', 'fenice')
        if os.path.exists(vendor_path):
            print(f"✅ FENICE vendor path exists: {vendor_path}")
        else:
            print(f"❌ FENICE vendor path not found: {vendor_path}")
            return False
            
        # Check key files exist
        key_files = ['__init__.py', 'FENICE.py']
        for file in key_files:
            file_path = os.path.join(vendor_path, file)
            if os.path.exists(file_path):
                print(f"✅ Key file exists: {file}")
            else:
                print(f"❌ Missing key file: {file}")
                return False
                
    except Exception as e:
        print(f"❌ Path check failed: {e}")
        return False
    
    return True

def test_fenice_scorer_import():
    """Test that FENICEScorer can be imported and configured."""
    print("\nTesting FENICEScorer import...")
    
    try:
        from rlvr_summary.rewards.fenice import create_fenice_scorer
        scorer = create_fenice_scorer(threshold=0.5)
        print("✅ FENICEScorer can be created")
        print(f"   - Threshold: {scorer.threshold}")
        print(f"   - Batch size: {scorer.batch_size}")
        print(f"   - Model loaded: {scorer._model_loaded}")
        return True
    except ImportError as e:
        print(f"❌ FENICEScorer import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ FENICEScorer creation failed: {e}")
        return False

def test_requirements_removed():
    """Test that FENICE was removed from requirements.txt."""
    print("\nTesting requirements.txt...")
    
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
            
        if 'FENICE' in content:
            print("❌ FENICE still found in requirements.txt")
            return False
        else:
            print("✅ FENICE successfully removed from requirements.txt")
            return True
            
    except Exception as e:
        print(f"❌ Could not check requirements.txt: {e}")
        return False

def test_dependencies_added():
    """Test that FENICE dependencies were added to pyproject.toml."""
    print("\nTesting pyproject.toml dependencies...")
    
    try:
        with open('pyproject.toml', 'r') as f:
            content = f.read()
            
        fenice_deps = ['fastcoref', 'sentencepiece']
        missing_deps = []
        
        for dep in fenice_deps:
            if dep not in content:
                missing_deps.append(dep)
                
        if missing_deps:
            print(f"❌ Missing FENICE dependencies: {missing_deps}")
            return False
        else:
            print("✅ FENICE dependencies added to pyproject.toml")
            print("   - fastcoref: found")
            print("   - sentencepiece: found")
            return True
            
    except Exception as e:
        print(f"❌ Could not check pyproject.toml: {e}")
        return False

def test_import_path_updates():
    """Test that import paths were updated correctly."""
    print("\nTesting import path updates...")
    
    files_to_check = [
        'src/rlvr_summary/rewards/fenice.py',
        'test_fenice_fix.py'
    ]
    
    success = True
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            if 'from metric.FENICE import FENICE' in content:
                print(f"❌ Old import path found in {file_path}")
                success = False
            elif 'from rlvr_summary.vendor.fenice import FENICE' in content:
                print(f"✅ Updated import path found in {file_path}")
            else:
                print(f"⚠️  No FENICE import found in {file_path}")
                
        except Exception as e:
            print(f"❌ Could not check {file_path}: {e}")
            success = False
            
    return success

if __name__ == "__main__":
    print("🧪 Testing FENICE integration structure...")
    print("=" * 60)
    
    tests = [
        test_import_structure,
        test_fenice_scorer_import,
        test_requirements_removed,
        test_dependencies_added,
        test_import_path_updates
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structural tests passed!")
        print("✅ FENICE has been successfully integrated as a vendor package")
        print("✅ Dependencies have been updated correctly")
        print("✅ Import paths have been fixed")
        print("\nℹ️  Note: Runtime tests will work once dependencies are installed")
        print("   Run 'pip install -e .' to install all dependencies")
    else:
        print("💥 Some tests failed. Please check the implementation.")
        sys.exit(1)
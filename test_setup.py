#!/usr/bin/env python3
"""
Test script to verify the Board Game NLP setup.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print("✅ Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import pandas
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import yaml
        print("✅ PyYAML imported successfully")
    except ImportError as e:
        print(f"❌ PyYAML import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading."""
    print("\n🔧 Testing configuration...")
    
    try:
        from boardgame_nlp.utils import ConfigManager
        config = ConfigManager()
        print("✅ Configuration loaded successfully")
        
        # Test some config values
        model_name = config.get('model.name')
        if model_name:
            print(f"✅ Model name: {model_name}")
        else:
            print("⚠️  Model name not found in config")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_directories():
    """Test if required directories exist or can be created."""
    print("\n📁 Testing directories...")
    
    required_dirs = ['data', 'Results', 'Model']
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ Directory exists: {dir_name}")
        else:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Directory created: {dir_name}")
            except Exception as e:
                print(f"❌ Failed to create directory {dir_name}: {e}")
                return False
    
    return True

def test_package_structure():
    """Test if the package structure is correct."""
    print("\n📦 Testing package structure...")
    
    required_files = [
        'boardgame_nlp/__init__.py',
        'boardgame_nlp/utils.py',
        'boardgame_nlp/data_collector.py',
        'boardgame_nlp/sentiment_analyzer.py',
        'boardgame_nlp/absa_analyzer.py',
        'boardgame_nlp/model_trainer.py',
        'config.yaml',
        'requirements.txt',
        'main.py'
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of the modules."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from boardgame_nlp.utils import ConfigManager, clean_text, is_english
        
        # Test config manager
        config = ConfigManager()
        print("✅ ConfigManager works")
        
        # Test text cleaning
        test_text = "This is a test http://example.com [IMG]text"
        cleaned = clean_text(test_text)
        if "http://example.com" not in cleaned and "[IMG]" not in cleaned:
            print("✅ Text cleaning works")
        else:
            print("❌ Text cleaning failed")
            return False
        
        # Test English detection
        english_result = is_english("This is English text")
        if english_result:
            print("✅ English detection works")
        else:
            print("❌ English detection failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Board Game NLP Setup Test")
    print("=" * 40)
    
    tests = [
        ("Package Structure", test_package_structure),
        ("Imports", test_imports),
        ("Directories", test_directories),
        ("Configuration", test_config),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

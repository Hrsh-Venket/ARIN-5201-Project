"""
Verification script to check if the project is set up correctly.
"""
import os
import sys


def check_file(path, description):
    """Check if a file exists."""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists


def check_directory(path, description):
    """Check if a directory exists."""
    exists = os.path.isdir(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists


def main():
    """Run all verification checks."""
    print("="*60)
    print("POSTER GENERATOR - SETUP VERIFICATION")
    print("="*60)

    all_good = True

    print("\n1. Core Files:")
    all_good &= check_file("main.py", "Main orchestration")
    all_good &= check_file("state.py", "State definition")
    all_good &= check_file("config.py", "Configuration")
    all_good &= check_file("requirements.txt", "Dependencies")

    print("\n2. Agent Files:")
    all_good &= check_file("agents/__init__.py", "Agent package init")
    all_good &= check_file("agents/planning_agent.py", "Planning Agent")
    all_good &= check_file("agents/text_generation_agent.py", "Text Generation Agent")
    all_good &= check_file("agents/editor_agent.py", "Editor Agent")
    all_good &= check_file("agents/image_generation_agent.py", "Image Generation Agent")
    all_good &= check_file("agents/text_adding_agent.py", "Text Adding Agent")

    print("\n3. Configuration Files:")
    all_good &= check_file(".env.example", "Environment template")
    env_exists = check_file(".env", "Environment variables")
    if not env_exists:
        print("  ⚠ You need to create .env from .env.example")
        all_good = False

    print("\n4. Output Directories:")
    all_good &= check_directory("outputs", "Output directory")
    all_good &= check_directory("intermediate_outputs", "Intermediate output directory")

    print("\n5. Input Files (Required for running):")
    input_text = check_file("input.txt", "Input text")
    input_image = check_file("input.png", "Input image")

    if not input_text:
        print("  ⚠ Create input.txt with your poster keywords")
        print("  Example: 'Tech Conference 2024, Innovation, AI'")

    if not input_image:
        print("  ⚠ Create input.png with your logo/mascot")

    print("\n6. Documentation:")
    check_file("README_IMPLEMENTATION.md", "Implementation guide")
    check_file("QUICKSTART.md", "Quick start guide")

    print("\n" + "="*60)
    if all_good and input_text and input_image:
        print("✓ ALL CHECKS PASSED - Ready to run!")
        print("  Run: python main.py")
    elif all_good:
        print("⚠ CORE SETUP COMPLETE - Need input files to run")
        print("  1. Create input.txt with poster keywords")
        print("  2. Create input.png with logo/mascot")
        print("  3. Ensure .env has valid API keys")
        print("  4. Run: python main.py")
    else:
        print("✗ SETUP INCOMPLETE - Fix issues above")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    main()

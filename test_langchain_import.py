import sys
import traceback

print("Testing langchain_coordinator import...")
print("=" * 60)

try:
    # Try importing just to see the error
    import langchain_coordinator
    print("SUCCESS: Import worked!")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

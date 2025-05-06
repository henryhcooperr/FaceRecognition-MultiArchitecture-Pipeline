#!/usr/bin/env python3

import unittest
import sys
import os

if __name__ == '__main__':
    # Add the parent directory to the path so unittest can find the tests
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    print("Running face recognition system tests...\n")
    
    # If specific tests were provided, run only those
    if len(sys.argv) > 1:
        test_names = sys.argv[1:]
        test_suite = unittest.TestLoader().loadTestsFromNames(test_names)
    else:
        # Otherwise, discover and run all tests
        test_suite = unittest.defaultTestLoader.discover('tests')
        
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    # Return non-zero exit code if tests failed
    sys.exit(not result.wasSuccessful()) 
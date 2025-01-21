# doctests.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""Run all doctests from modules on the command line.  Use -v for verbose.

Example usages:

    python doctests.py *.py
    python doctests.py -v *.py

You can add more module-level tests with
    __doc__ += "..."
You can add stochastic tests with
    __doc__ += random_tests("...")
"""

if __name__ == "__main__":
    import sys, glob, doctest
    args = [arg for arg in sys.argv[1:] if arg != '-v']
    if not args: args = ['*.py']
    modules = [__import__(name.replace('.py',''))
               for arg in args for name in glob.glob(arg)]

    print("Testing %d modules..." % len(modules))
    for module in modules:
        doctest.testmod(module, report=1, optionflags=doctest.REPORT_UDIFF)
    summary = doctest.master.summarize() if modules else (0, 0)
    
    print()
    print()
    print('%d failed out of %d tests' % summary)

# logic.py
# --------
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


"""Representations and Inference for the CS 188 Logic Project

Code originally from https://code.google.com/p/aima-python/
Modified heavily with additional convenience classes and functions as well
as an interface to the pycosat (picoSAT wrapper) library.
https://pypi.python.org/pypi/pycosat.
Original package contained implementations of functions and data structures
for Knowledge bases and First-Order Logic.
"""

import itertools, re
from typing import Tuple
import agents
from logic_utils import *
import pycosat

#______________________________________________________________________________

class Expr:
    """A symbolic mathematical expression.  We use this class for logical
    expressions, and for terms within logical expressions. In general, an
    Expr has an op (operator) and a list of args.  The op can be:
      Null-ary (no args) op:
        A number, representing the number itself.  (e.g. Expr(42) => 42)
        A symbol, representing a variable or constant (e.g. Expr('F') => F)
      Unary (1 arg) op:
        '~', '-', representing NOT, negation (e.g. Expr('~', Expr('P')) => ~P)
      Binary (2 arg) op:
        '>>', '<<', representing forward and backward implication
        '+', '-', '*', '/', '**', representing arithmetic operators
        '<', '>', '>=', '<=', representing comparison operators
        '<=>', '^', representing logical equality and XOR
      N-ary (0 or more args) op:
        '&', '|', representing conjunction and disjunction
        A symbol, representing a function term or FOL proposition

    Exprs can be constructed with operator overloading: if x and y are Exprs,
    then so are x + y and x & y, etc.  Also, if F and x are Exprs, then so is
    F(x); it works by overloading the __call__ method of the Expr F.  Note
    that in the Expr that is created by F(x), the op is the str 'F', not the
    Expr F.   See http://www.python.org/doc/current/ref/specialnames.html
    to learn more about operator overloading in Python.

    WARNING: x == y and x != y are NOT Exprs.  The reason is that we want
    to write code that tests 'if x == y:' and if x == y were the same
    as Expr('==', x, y), then the result would always be true; not what a
    programmer would expect.  But we still need to form Exprs representing
    equalities and disequalities.  We concentrate on logical equality (or
    equivalence) and logical disequality (or XOR).  You have 3 choices:
        (1) Expr('<=>', x, y) and Expr('^', x, y)
            Note that ^ is bitwise XOR in Python (and Java and C++)
        (2) expr('x <=> y') and expr('x =/= y').
            See the doc string for the function expr.
        (3) (x % y) and (x ^ y).
            It is very ugly to have (x % y) mean (x <=> y), but we need
            SOME operator to make (2) work, and this seems the best choice.

    WARNING: if x is an Expr, then so is x + 1, because the int 1 gets
    coerced to an Expr by the constructor.  But 1 + x is an error, because
    1 doesn't know how to add an Expr.  (Adding an __radd__ method to Expr
    wouldn't help, because int.__add__ is still called first.) Therefore,
    you should use Expr(1) + x instead, or ONE + x, or expr('1 + x').
    """
    
    # Initialize a counter object
    counter = 0
    def __init__(self, op, *args): 
        "Op is a string or number; args are Exprs (or are coerced to Exprs)."
        assert isinstance(op, str) or (isnumber(op) and not args)
        self.op = num_or_str(op)
        self.args = tuple(map(expr, args)) ## Coerce args to Exprs
        if not args and not is_prop_symbol(self.op):
            raise SyntaxError("Unacceptable symbol base name (%s). Name must start with an upper-case alphabetic character that and is not TRUE or FALSE. Furthermore, only the following are allowed: capital and lower case alphabetic, 0-9, _, \",\", [, and ]." % self.op)
        # Increment the counter when an object is created
        type(self).counter += 1
        
    def __call__(self, *args):
        """Self must be a symbol with no args, such as Expr('F').  Create a new
        Expr with 'F' as op and the args as arguments."""
        assert is_symbol(self.op) and not self.args
        return Expr(self.op, *args)

    def __repr__(self):
        "Show something like 'P' or 'P(x, y)', or '~P' or '(P | Q | R)'"
        if not self.args:         # Constant or proposition with arity 0
            return str(self.op)
        elif is_symbol(self.op):  # Functional or propositional operator
            return '%s(%s)' % (self.op, ', '.join(map(repr, self.args)))
        elif len(self.args) == 1: # Prefix operator
            return self.op + repr(self.args[0])
        else:                     # Infix operator
            return '(%s)' % (' '+self.op+' ').join(map(repr, self.args))

    def __eq__(self, other):
        """x and y are equal iff their ops and args are equal."""
        return (other is self) or (isinstance(other, Expr)
            and self.op == other.op and self.args == other.args)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        "Need a hash method so Exprs can live in dicts."
        return hash(self.op) ^ hash(tuple(self.args))

    # See http://www.python.org/doc/current/lib/module-operator.html
    # Not implemented: not, abs, pos, concat, contains, *item, *slice
    def __lt__(self, other):     return Expr('<',  self, other)
    def __le__(self, other):     return Expr('<=', self, other)
    def __ge__(self, other):     return Expr('>=', self, other)
    def __gt__(self, other):     return Expr('>',  self, other)
    def __add__(self, other):    return Expr('+',  self, other)
    def __sub__(self, other):    return Expr('-',  self, other)
    def __and__(self, other):    return Expr('&',  self, other)
    def __div__(self, other):    return Expr('/',  self, other)
    def __truediv__(self, other):return Expr('/',  self, other)
    def __invert__(self):        return Expr('~',  self)
    def __lshift__(self, other): return Expr('<<', self, other)
    def __rshift__(self, other): return Expr('>>', self, other)
    def __mul__(self, other):    return Expr('*',  self, other)
    def __neg__(self):           return Expr('-',  self)
    def __or__(self, other):     return Expr('|',  self, other)
    def __pow__(self, other):    return Expr('**', self, other)
    def __xor__(self, other):    return Expr('^',  self, other)
    def __mod__(self, other):    return Expr('<=>',  self, other)

class PropSymbolExpr(Expr):
    """An extension of Expr intended to represent a symbol. This SymbolExpr
    is a convenience for naming symbols, especially symbols whose names
    indicate an indexed value (e.g. Position[x,y] or Fluent[t]).
    Symbol name must begin with a capital letter. This class helps to add
    brackets with enumerated indices to the end of the name.
    """
    # copied from logicPlan.py; preferably do this better
    pacman_str = 'P'
    food_str = 'FOOD'
    wall_str = 'WALL'
    DIRECTIONS = {'North', 'South', 'East', 'West'}
    # rules
    double_index = {pacman_str, food_str, wall_str}
    time_index = {pacman_str, food_str} | DIRECTIONS
    all_checked = double_index | time_index

    def __init__(self, sym_str: str, *index: Tuple[int], time: int = None):
        """Constructor taking a propositional logic symbol name and an optional set of index values,
        creating a symbol with the base name followed by brackets with the specific
        indices.
        sym_str: String representing base name for symbol. Must begin with a capital letter.
        Examples:
        >>> red = PropSymbolExpr("R")
        >>> print(red)
        R
        >>> turnLeft7 = PropSymbolExpr("Left",7)
        >>> print(turnLeft7)
        Left[7]
        >>> pos_2_3 = PropSymbolExpr("P",2,3)
        >>> print(pos_2_3)
        P[2,3]
        """
        if not is_prop_symbol(sym_str):
            raise SyntaxError("Unacceptable symbol base name (%s). Name must start with an upper-case alphabetic character that and is not TRUE or FALSE. Furthermore, only the following are allowed: capital and lower case alphabetic, 0-9, _, \",\", [, and ]." % sym_str)
        if sym_str in self.all_checked:
            if sym_str in self.double_index:
                if len(index) != 2:
                    raise SyntaxError("Unexpected " + sym_str + " Symbol. Was expecting 2 coordinates.")
            elif len(index) != 0:
                raise SyntaxError("Unexpected " + sym_str + " Symbol. Was expecting 0 coordinates.")
            if sym_str in self.time_index:
                if time == None:
                    raise SyntaxError("Unexpected " + sym_str + " Symbol. Was expecting time stamp.")
            elif time != None:
                raise SyntaxError("Unexpected " + sym_str + " Symbol. Was expecting no time stamp.")
        self.sym_str = sym_str
        self.indicies = index
        self.time = time
        if len(index) > 0:
            if len(index) > 4:
                raise SyntaxError("Too many arguments to SymbolExpr constructor. SymbolExpr(symbol_str, [index1], [index2], [index3], [index4], time=[time]), or fewer indicies -- possibly 0.")
            if len(index) == 1:
                sym_str = '%s[%d]' % (sym_str, *index)
            elif len(index) == 2:
                sym_str = '%s[%d,%d]' % (sym_str, *index)
            elif len(index) == 3:
                sym_str = '%s[%d,%d,%d]' % (sym_str, *index)
            elif len(index) == 4:
                sym_str = '%s[%d,%d,%d,%d]' % (sym_str, *index)
        if time != None:
            sym_str = '%s_%d' % (sym_str, int(time))
        Expr.__init__(self, sym_str)
        
    def getBaseName(self):
        return self.sym_str
    
    def getIndex(self):
        return self.indicies
    
    def getTime(self):
        return self.time
    
def parseExpr(symbol):
    """A simple expression parser, takes in a PropSymbolExpr and returns 
       its deconstruction in the form ( sym_str, indices, time ).
       Examples:
       >>> parseExpr("North[3]")
       ('North', None, (3))
       >>> parseExpr("A")
       (A, None, ())
       >>> parseExpr("P[3,4]_1")
       ('P', 1, (3, 4))
    """
    tokens = re.split(r"_", str(symbol))
    time = None
    if len(tokens) == 2:
        symbol = tokens[0]
        time = int(tokens[1])

    tokens = re.findall(r"[\w]+", str(symbol))
    if len(tokens) == 1:
        return (tokens[0], (), time)
    return (tokens[0], tuple(map(int,tokens[1:])), time)

def expr(s):
    """Create an Expr representing a logic expression by parsing the input
    string. Symbols and numbers are automatically converted to Exprs.
    In addition you can use alternative spellings of these operators:
      'x ==> y'   parses as   (x >> y)    # Implication
      'x <== y'   parses as   (x << y)    # Reverse implication
      'x <=> y'   parses as   (x % y)     # Logical equivalence
      'x =/= y'   parses as   (x ^ y)     # Logical disequality (xor)
    But BE CAREFUL; precedence of implication is wrong. expr('P & Q ==> R & S')
    is ((P & (Q >> R)) & S); so you must use expr('(P & Q) ==> (R & S)').
    >>> expr('P <=> Q(1)')
    (P <=> Q(1))
    >>> expr('P & Q | ~R(x, F(x))')
    ((P & Q) | ~R(x, F(x)))
    """
    if isinstance(s, Expr): return s
    if isnumber(s): return Expr(s)
    ## Replace the alternative spellings of operators with canonical spellings
    s = s.replace('==>', '>>').replace('<==', '<<')
    s = s.replace('<=>', '%').replace('=/=', '^')
    ## Replace a symbol or number, such as 'P' with 'Expr("P")'
    s = re.sub(r'([a-zA-Z0-9_.]+)', r'Expr("\1")', s)
    ## Now eval the string.  (A security hole; do not use with an adversary.)
    return eval(s, {'Expr':Expr})

def is_symbol(s):
    "A string s is a symbol if it starts with an alphabetic char."
    return isinstance(s, str) and s[:1].isalpha()

def is_var_symbol(s):
    "A logic variable symbol is an initial-lowercase string."
    return is_symbol(s) and s[0].islower()

def is_prop_symbol(s):
    """A proposition logic symbol is an initial-uppercase string other than
    TRUE or FALSE."""
    return is_symbol(s) and s[0].isupper() and s != 'TRUE' and s != 'FALSE' and re.match(r'[a-zA-Z0-9_\[\],]*$', s)

def variables(s):
    """Return a set of the variables in expression s.
    >>> ppset(variables(F(x, A, y)))
    set([x, y])
    >>> ppset(variables(F(G(x), z)))
    set([x, z])
    >>> ppset(variables(expr('F(x, x) & G(x, y) & H(y, z) & R(A, z, z)')))
    set([x, y, z])
    """
    result = set([])
    def walk(s):
        if is_variable(s):
            result.add(s)
        else:
            for arg in s.args:
                walk(arg)
    walk(s)
    return result

def is_definite_clause(s):
    """returns True for exprs s of the form A & B & ... & C ==> D,
    where all literals are positive.  In clause form, this is
    ~A | ~B | ... | ~C | D, where exactly one clause is positive.
    >>> is_definite_clause(expr('Farmer(Mac)'))
    True
    >>> is_definite_clause(expr('~Farmer(Mac)'))
    False
    >>> is_definite_clause(expr('(Farmer(f) & Rabbit(r)) ==> Hates(f, r)'))
    True
    >>> is_definite_clause(expr('(Farmer(f) & ~Rabbit(r)) ==> Hates(f, r)'))
    False
    >>> is_definite_clause(expr('(Farmer(f) | Rabbit(r)) ==> Hates(f, r)'))
    False
    """
    if is_symbol(s.op):
        return True
    elif s.op == '>>':
        antecedent, consequent = s.args
        return (is_symbol(consequent.op)
                and every(lambda arg: is_symbol(arg.op), conjuncts(antecedent)))
    else:
        return False

def parse_definite_clause(s):
    "Return the antecedents and the consequent of a definite clause."
    assert is_definite_clause(s)
    if is_symbol(s.op):
        return [], s
    else:
        antecedent, consequent = s.args
        return conjuncts(antecedent), consequent

## Useful constant Exprs used in examples and code:
class SpecialExpr(Expr):
    """Exists solely to allow the normal Expr constructor to assert valid symbol
    syntax while still having some way to create the constants 
    TRUE, FALSE, ZERO, ONE, and, TWO
    """
    def __init__(self, op, *args):
        "Op is a string or number; args are Exprs (or are coerced to Exprs)."
        assert isinstance(op, str) or (isnumber(op) and not args)
        self.op = num_or_str(op)
        self.args = tuple(map(expr, args)) ## Coerce args to Exprs

TRUE, FALSE = tuple(map(SpecialExpr, ['TRUE', 'FALSE']))
ZERO, ONE, TWO = tuple(map(SpecialExpr, [0, 1, 2]))
A, B, C, D, E, F, G, P, Q  = tuple(map(Expr, 'ABCDEFGPQ'))

#______________________________________________________________________________
def prop_symbols(x):
    "Return a list of all propositional symbols in x."
    if not isinstance(x, Expr):
        return []
    elif is_prop_symbol(x.op):
        return [x]
    else:
        return list(set(symbol for arg in x.args
                        for symbol in prop_symbols(arg)))

def pl_true(exp, model={}):
    """Return True if the propositional logic expression is true in the model,
    and False if it is false. If the model does not specify the value for
    every proposition, this may return None to indicate 'not obvious';
    this may happen even when the expression is tautological."""
    op, args = exp.op, exp.args
    if exp == TRUE:
        return True
    elif exp == FALSE:
        return False
    elif is_prop_symbol(op):
        return model.get(exp)
    elif op == '~':
        p = pl_true(args[0], model)
        if p is None: return None
        else: return not p
    elif op == '|':
        result = False
        for arg in args:
            p = pl_true(arg, model)
            if p is True: return True
            if p is None: result = None
        return result
    elif op == '&':
        result = True
        for arg in args:
            p = pl_true(arg, model)
            if p is False: return False
            if p is None: result = None
        return result
    p, q = args
    if op == '>>':
        return pl_true(~p | q, model)
    elif op == '<<':
        return pl_true(p | ~q, model)
    pt = pl_true(p, model)
    if pt is None: return None
    qt = pl_true(q, model)
    if qt is None: return None
    if op == '<=>':
        return pt == qt
    elif op == '^':
        return pt != qt
    else:
        raise ValueError("illegal operator in logic expression" + str(exp))

#______________________________________________________________________________

## Convert to Conjunctive Normal Form (CNF)

def to_cnf(s):
    """Convert a propositional logical sentence s to conjunctive normal form.
    That is, to the form ((A | ~B | ...) & (B | C | ...) & ...) [p. 253]
    >>> to_cnf("~(B|C)")
    (~B & ~C)
    >>> to_cnf("B <=> (P1|P2)")
    ((~P1 | B) & (~P2 | B) & (P1 | P2 | ~B))
    >>> to_cnf("a | (b & c) | d")
    ((b | a | d) & (c | a | d))
    >>> to_cnf("A & (B | (D & E))")
    (A & (D | B) & (E | B))
    >>> to_cnf("A | (B | (C | (D & E)))")
    ((D | A | B | C) & (E | A | B | C))
    """
    if isinstance(s, str): s = expr(s)
    s = eliminate_implications(s) # Steps 1, 2 from p. 253
    s = move_not_inwards(s) # Step 3
    s = distribute_and_over_or(s) # Step 4
    return s

def eliminate_implications(s):
    """Change >>, <<, and <=> into &, |, and ~. That is, return an Expr
    that is equivalent to s, but has only &, |, and ~ as logical operators.
    >>> eliminate_implications(A >> (~B << C))
    ((~B | ~C) | ~A)
    >>> eliminate_implications(A ^ B)
    ((A & ~B) | (~A & B))
    """
    if not s.args or is_symbol(s.op): return s     ## (Atoms are unchanged.)
    args = tuple(map(eliminate_implications, s.args))
    a, b = args[0], args[-1]
    if s.op == '>>':
        return (b | ~a)
    elif s.op == '<<':
        return (a | ~b)
    elif s.op == '<=>':
        return (a | ~b) & (b | ~a)
    elif s.op == '^':
        assert len(args) == 2   ## TODO: relax this restriction
        return (a & ~b) | (~a & b)
    else:
        assert s.op in ('&', '|', '~')
        return Expr(s.op, *args)

def move_not_inwards(s):
    """Rewrite sentence s by moving negation sign inward.
    >>> move_not_inwards(~(A | B))
    (~A & ~B)
    >>> move_not_inwards(~(A & B))
    (~A | ~B)
    >>> move_not_inwards(~(~(A | ~B) | ~~C))
    ((A | ~B) & ~C)
    """
    if s.op == '~':
        NOT = lambda b: move_not_inwards(~b)
        a = s.args[0]
        if a.op == '~': return move_not_inwards(a.args[0]) # ~~A ==> A
        if a.op =='&': return associate('|', tuple(map(NOT, a.args)))
        if a.op =='|': return associate('&', tuple(map(NOT, a.args)))
        return s
    elif is_symbol(s.op) or not s.args:
        return s
    else:
        return Expr(s.op, *map(move_not_inwards, s.args))

def distribute_and_over_or(s):
    """Given a sentence s consisting of conjunctions and disjunctions
    of literals, return an equivalent sentence in CNF.
    >>> distribute_and_over_or((A & B) | C)
    ((A | C) & (B | C))
    """
    if s.op == '|':
        s = associate('|', s.args)
        if s.op != '|':
            return distribute_and_over_or(s)
        if len(s.args) == 0:
            return FALSE
        if len(s.args) == 1:
            return distribute_and_over_or(s.args[0])
        conj = find_if((lambda d: d.op == '&'), s.args)
        if not conj:
            return s
        others = [a for a in s.args if a is not conj]
        rest = associate('|', others)
        return associate('&', [distribute_and_over_or(c|rest)
                               for c in conj.args])
    elif s.op == '&':
        return associate('&', map(distribute_and_over_or, s.args))
    else:
        return s

def associate(op, args):
    """Given an associative op, return an expression with the same
    meaning as Expr(op, *args), but flattened -- that is, with nested
    instances of the same op promoted to the top level.
    >>> associate('&', [(A&B),(B|C),(B&C)])
    (A & B & (B | C) & B & C)
    >>> associate('|', [A|(B|(C|(A&B)))])
    (A | B | C | (A & B))
    """
    args = dissociate(op, args)
    if len(args) == 0:
        return _op_identity[op]
    elif len(args) == 1:
        return args[0]
    else:
        return Expr(op, *args)

_op_identity = {'&':TRUE, '|':FALSE, '+':ZERO, '*':ONE}

def conjoin(exprs, *args):
    """Given a list of expressions, returns their conjunction. Can be called either
    with one argument that is a list of expressions, or with several arguments that
    are each an expression.
    If exprs is a singular expression or contains only one expression, return that
    expression directly.
    If exprs is an empty list, throw an error.
    >>> conjoin([(A&B),(B|C),(B&C)])
    (A & B & (B | C) & B & C)
    >>> conjoin((A&B), (B|C), (B&C))
    (A & B & (B | C) & B & C)
    >>> conjoin([A])
    A
    """
    if args:
        return conjoin([exprs] + list(args))
    if (type(exprs) != list):
        return exprs

    assert len(exprs) > 0, "List to conjoin cannot be empty."

    # It is a list. Enforce everything in the list is an Expr
    for expr in exprs:
        assert isinstance(expr, Expr), "An item in list to conjoin is not an Expr."

    if (len(exprs) == 1):
        return exprs[0]
    return associate('&', exprs)

def disjoin(exprs, *args):
    """Given a list of expressions, returns their disjunction. Can be called either
    with one argument that is a list of expressions, or with several arguments that
    are each an expression.
    If exprs is a singular expression or contains only one expression, return that
    expression directly.
    If exprs is an empty list, throw an error.
    >>> disjoin([C, (A&B), (D&E)])
    (C | (A & B) | (D & E))
    >>> disjoin(C, (A&B), (D&E))
    (C | (A & B) | (D & E))
    >>> disjoin([C])
    D
    """
    if args:
        return disjoin([exprs] + list(args))
    if (type(exprs) != list):
        return exprs

    assert len(exprs) > 0, "List to disjoin cannot be empty."

    # It is a list. Enforce everything in the list is an Expr
    for expr in exprs:
        assert isinstance(expr, Expr), "An item in list to disjoin is not an Expr."

    if (len(exprs) == 1):
        return exprs[0]
    return associate('|', exprs)

def dissociate(op, args):
    """Given an associative op, return a flattened list result such
    that Expr(op, *result) means the same as Expr(op, *args)."""
    result = []
    def collect(subargs):
        for arg in subargs:
            if arg.op == op: collect(arg.args)
            else: result.append(arg)
    collect(args)
    return result

def conjuncts(s):
    """Return a list of the conjuncts in the sentence s.
    >>> conjuncts(A & B)
    [A, B]
    >>> conjuncts(A | B)
    [(A | B)]
    """
    return dissociate('&', [s])

def disjuncts(s):
    """Return a list of the disjuncts in the sentence s.
    >>> disjuncts(A | B)
    [A, B]
    >>> disjuncts(A & B)
    [(A & B)]
    """
    return dissociate('|', [s])

def is_valid_cnf(exp):
    if not isinstance(exp, Expr):
        print("Input is not an expression.")
        return False
    
    clauses = conjuncts(exp);
    
    for c in clauses:
        literals = disjuncts(c)
        
        for lit in literals:
            if len(lit.args) == 0:
                symbol = lit;
            elif len(lit.args) == 1:
                symbol = lit.args[0]
                
                if len(symbol.args) != 0:
                    print("Found a NOT outside of %s" % symbol)
                    return False
                
            else:
                print("Found %s where only a literal should be." % lit)
                return False
    
            symbol_str = str(symbol)
    
            if not is_symbol(symbol_str):
                print("%s is not a valid symbol." % symbol_str)
                return False
            elif not symbol_str[0].isupper():
                print("The symbol %s must begin with an upper-case letter." % symbol_str)
                return False
            elif symbol_str == 'TRUE':
                print("TRUE is not a valid symbol.")
                return False
            elif symbol_str == 'FALSE':
                print("FALSE is not a valid symbol.")
                return False
        
    return True

#______________________________________________________________________________
# pycosat python wrapper around PicoSAT software.
# https://pypi.python.org/pypi/pycosat

def pycoSAT(expr):
    """Check satisfiability of an expression.
    Given a CNF expression, returns a model that causes the input expression
    to be true. Returns false if it cannot find a satisfible model.
    A model is simply a dictionary with Expr symbols as keys with corresponding values
    that are booleans: True if that symbol is true in the model and False if it is
    false in the model.
    Calls the pycosat solver: https://pypi.python.org/pypi/pycosat
    >>> ppsubst(pycoSAT(A&~B))
    {A: True, B: False}
    >>> pycoSAT(P&~P)
    False
    """

    clauses = conjuncts(expr)

    # Load symbol dictionary
    symbol_dict = mapSymbolAndIndices(clauses)
    # Convert Expr to integers
    clauses_int = exprClausesToIndexClauses(clauses, symbol_dict)
    
    model_int = pycosat.solve(clauses_int)
    
    if model_int == 'UNSAT' or model_int == 'UNKNOWN':
        return False
    
    model = indexModelToExprModel(model_int, symbol_dict)
    
    return model

def mapSymbolAndIndices(clauses):
    """
    Create a dictionary that maps each clause to an integer index.
    Uses a bidirectional dictionary {key1:value1, value1:key1, ...} for quick
    access from symbol to index and index to symbol.
    """
    symbol_dict = {}
    idx = 1
    for clause in clauses:
        symbols = prop_symbols(clause)
        for symbol in symbols:
            if symbol not in symbol_dict:
                symbol_dict[symbol] = idx
                symbol_dict[idx] = symbol
                idx +=1

    return symbol_dict

def exprClausesToIndexClauses(clauses, symbol_dict):
    """
    Convert each Expr in a list of clauses (CNF) into its corresponding index in
    the symbol_dict (see mapSymbolAndIndices) 
    """
    clauses_int = []
    for c in clauses:
        c_disj = disjuncts(c)
            
        c_int = []
        for lit in c_disj:
            # If literal is symbol, convert to index and add it.
            # Otherwise it is ~symbol, in which case, we extract the symbol, 
            # convert it to index, and add the negative of the index
            if len(lit.args) == 0:
                c_int += [symbol_dict[lit]]
            else:
                c_int += [-symbol_dict[lit.args[0]]]
        clauses_int += [c_int]

    return clauses_int

def indexModelToExprModel(model_int, symbol_dict):
    """
    Convert a model with indices into a model with the corresponding Expr in
    the symbol_dict (see mapSymbolAndIndices)
    >>>
    """
    model = {}
    for lit_int in model_int:
        if lit_int > 0:
            model[symbol_dict[lit_int]] = True
        else:
            model[symbol_dict[-lit_int]] = False
            
    return model

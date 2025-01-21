# logic_planTestClasses.py
# ------------------------
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


import testClasses

import textDisplay
import graphicsDisplay
import layout
import pacman
import logicAgents 
from logicPlan import PlanningProblem
import logicPlan

import itertools

# Simple test case which evals an arbitrary piece of python code.
# The test is correct if the output of the code given the student's
# solution matches that of the instructor's.
        

class EvalTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(EvalTest, self).__init__(question, testDict)
        self.preamble = compile(testDict.get('preamble', ""), "%s.preamble" % self.getPath(), 'exec')
        self.test = compile(testDict['test'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']

    def evalCode(self, moduleDict):
        bindings = dict(moduleDict)
        exec(self.preamble, bindings)
        return str(eval(self.test, bindings))

    def execute(self, grades, moduleDict, solutionDict):
        result = self.evalCode(moduleDict)
        if result == solutionDict['result']:
            grades.addMessage('PASS: %s' % self.path)
            grades.addMessage('\t%s' % self.success)
            return True
        else:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\tstudent result: "%s"' % result)
            grades.addMessage('\tcorrect result: "%s"' % solutionDict['result'])

        return False

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# The result of evaluating the test must equal the below when cast to a string.\n')

        handle.write('result: "%s"\n' % self.evalCode(moduleDict))
        handle.close()
        return True

    # BEGIN SOLUTION NO PROMPT
    def createPublicVersion(self):
        pass
    # END SOLUTION NO PROMPT
        
class LogicTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(LogicTest, self).__init__(question, testDict)
        self.preamble = compile(testDict.get('preamble', ""), "%s.preamble" % self.getPath(), 'exec')
        self.test = compile(testDict['test'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']

    def evalCode(self, moduleDict):
        bindings = dict(moduleDict)
        exec(self.preamble, bindings)
        return eval(self.test, bindings)

    def execute(self, grades, moduleDict, solutionDict):
        result = self.evalCode(moduleDict)
        result = map(lambda x: str(x), result)
        result = ' '.join(result)
        
        if result == solutionDict['result']:
            grades.addMessage('PASS: %s' % self.path)
            grades.addMessage('\t%s' % self.success)
            return True
        for i in range(100):
            solI = 'result' + str(i)
            if solI not in solutionDict:
                break
            if result == solutionDict[solI]:
                grades.addMessage('PASS: %s' % self.path)
                grades.addMessage('\t%s' % self.success)
                return True
        grades.addMessage('FAIL: %s' % self.path)
        grades.addMessage('\t%s' % self.failure)
        grades.addMessage('\tstudent result: "%s"' % result)
        grades.addMessage('\tcorrect result: "%s"' % solutionDict['result'])

        return False

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# The result of evaluating the test must equal the below when cast to a string.\n')
        solution = self.evalCode(moduleDict)
        solution = map(lambda x: str(x), solution)
        handle.write('result: "%s"\n' % ' '.join(solution))
        handle.close()
        return True        

    # BEGIN SOLUTION NO PROMPT
    def createPublicVersion(self):
        pass
    # END SOLUTION NO PROMPT

class PacphysicsTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(PacphysicsTest, self).__init__(question, testDict)
        self.layoutText = testDict['layout']
        self.layoutName = testDict['layoutName']
        self.t = int(testDict['t'])
        self.soln_labels = ["pacphysicsAxioms"]
        self.axiom_type = testDict['axiomType']
        if self.axiom_type == 'sensor':
            self.sensorAxioms = logicPlan.sensorAxioms
            self.successorAxioms = logicPlan.allLegalSuccessorAxioms
        elif self.axiom_type == 'slam':
            self.sensorAxioms = logicPlan.SLAMSensorAxioms
            self.successorAxioms = logicPlan.SLAMSuccessorAxioms
        else:
            raise Exception('Bad test case!')

    def solution(self, logicPlan):
        lay = layout.Layout([l.strip() for l in self.layoutText.split('\n')])
        walls_list = lay.walls.data
        all_coords = lay.get_all_coords_list()
        non_outer_wall_coords = lay.get_non_outer_wall_coords_list()
        pacphysics_axioms = logicPlan.pacphysicsAxioms(self.t, all_coords, non_outer_wall_coords, walls_list, self.sensorAxioms, self.successorAxioms)
        return pacphysics_axioms

    def execute(self, grades, moduleDict, solutionDict):
        grades.addMessage('Testing pacphysicsAxioms')
        logicPlan = moduleDict['logicPlan']
        gold_solution = solutionDict[self.soln_labels[0]]

        solution = self.solution(logicPlan)

        gold_soln_clauses_list_being_conjoined = str(gold_solution)[1:-1].split(" & ")
        soln_clauses_list_being_conjoined = str(solution)[1:-1].split(" & ")

        # Check student used conjoin correctly; this is a weak check
        # after <=>, we get Action) | (Wall expresisons due to SLAM successor
        for soln_clause in soln_clauses_list_being_conjoined:
            if "<=>" in soln_clause:
                if self.axiom_type == 'sensor':
                    continue
                else:
                    break
            contains_open_parens = ("(" in soln_clause[1:-1]) or ("(" in soln_clause[1:-1])
            if contains_open_parens:
                grades.addMessage('FAIL: {}'.format(self.path))
                grades.addMessage('\tStudent solution does not combine sentences properly.')
                grades.addMessage('\tMake sure you append the items to join with "and",'
                    ' and conjoin at the end.')
                return False

        # Check number of clauses is correct.
        gold_soln_num_clauses_conjoined = len(gold_soln_clauses_list_being_conjoined)
        soln_num_clauses_conjoined = len(soln_clauses_list_being_conjoined)

        if gold_soln_num_clauses_conjoined != soln_num_clauses_conjoined:
            grades.addMessage('FAIL: {}'.format(self.path))
            grades.addMessage('\tStudent solution differed from autograder solution')
            grades.addMessage('\tNumber of clauses being conjoined in student solution: {}'.format(
                soln_num_clauses_conjoined))
            grades.addMessage('\tNumber of clauses being conjoined in correct solution: {}'.format(
                gold_soln_num_clauses_conjoined))
            return False

        for gold_clause in gold_soln_clauses_list_being_conjoined:
            if gold_clause not in soln_clauses_list_being_conjoined:
                grades.addMessage('FAIL: {}'.format(self.path))
                grades.addMessage('\tStudent solution does not contain clause {}'.format(gold_clause))
                return False

        if set(soln_clauses_list_being_conjoined) != set(gold_soln_clauses_list_being_conjoined):
            grades.addMessage('FAIL: {}'.format(self.path))
            grades.addMessage('\tStudent solution differed from autograder solution on clause set comparison')
            grades.addMessage('\tStudent solution: {}'.format(solution))
            grades.addMessage('\tCorrect solution: {}'.format(gold_solution))
            return False
        
        if sorted(str(solution)) != sorted(str(gold_solution)):
            grades.addMessage('FAIL: {}'.format(self.path))
            grades.addMessage('\tStudent solution differed from autograder solution on character list comparison')
            grades.addMessage('\tStudent solution: {}'.format(solution))
            grades.addMessage('\tCorrect solution: {}'.format(gold_solution))
            return False

        grades.addMessage('PASS: %s' % self.path)
        return True

    def writeSolution(self, moduleDict, filePath):
        logicPlan = moduleDict['logicPlan']
        # open file and write comments
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        print("Solving problem", self.layoutName)
        print(self.layoutText)

        solution = self.solution(logicPlan)

        print("Problem solved")

        handle.write('{}: "{}"\n'.format(self.soln_labels[0], str(solution)))
        handle.close()

    # BEGIN SOLUTION NO PROMPT
    def createPublicVersion(self):
        pass
    # END SOLUTION NO PROMPT

class LocationSatisfiabilityTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(LocationSatisfiabilityTest, self).__init__(question, testDict)
        self.layoutText = testDict['layout']
        self.layoutName = testDict['layoutName']
        self.x0_y0 = eval(testDict['x0_y0'])
        self.action0 = testDict['action0']
        self.x1_y1 = eval(testDict['x1_y1'])
        self.action1 = testDict['action1']
        self.soln_labels = ["model_at_x1_y1_1", "model_not_at_x1_y1_1"]

    def solution(self, logicPlan):
        lay = layout.Layout([l.strip() for l in self.layoutText.split('\n')])
        pac = logicAgents.CheckSatisfiabilityAgent('checkLocationSatisfiability', 'LocMapProblem', logicPlan)
        ghosts = []
        disp = textDisplay.NullGraphics()
        games = next(pacman.runGames(lay, pac, ghosts, disp, 1, False, catchExceptions=True, timeout=180))
        loc_sat_models = logicPlan.checkLocationSatisfiability(self.x1_y1, self.x0_y0, self.action0, self.action1, pac.problem)
        return loc_sat_models

    def execute(self, grades, moduleDict, solutionDict):
        grades.addMessage('Testing checkLocationSatisfiability')
        logicPlan = moduleDict['logicPlan']

        solution = self.solution(logicPlan)

        for i, solution_i in enumerate(solution):
            gold_solution_i = solutionDict[self.soln_labels[i]]
            solution_i = logicPlan.modelToString(solution_i)

            if gold_solution_i == "False" and solution_i != "False":
                grades.addMessage('FAIL: {}'.format(self.path))
                grades.addMessage('\tStudent solution differed from autograder solution for {}'.format(self.soln_labels[i]))
                grades.addMessage('\tStudent model found satisfiable solution but no satisfiable solution exists.')
                return False
            elif gold_solution_i != "False" and solution_i == "False":
                grades.addMessage('FAIL: {}'.format(self.path))
                grades.addMessage('\tStudent solution differed from autograder solution for {}'.format(self.soln_labels[i]))
                grades.addMessage('\tStudent model found no satisfiable solution when a satisfiable solution exists.')
                return False
            elif gold_solution_i == "False" and solution_i == "False":
                continue
            else:
                pass

            gold_solution_i_str_pairs_list = gold_solution_i[2:-2].split("), (")
            gold_solution_i_tuples_list = [tuple(pair.split(", ")) for pair in gold_solution_i_str_pairs_list]
            gold_solution_i_dict = dict(gold_solution_i_tuples_list)
            solution_i_str_pairs_list = solution_i[2:-2].split("), (")
            solution_i_tuples_list = [tuple(pair.split(", ")) for pair in solution_i_str_pairs_list]
            solution_i_dict = dict(solution_i_tuples_list)

            # Check if student has all of the correct variables.
            gold_solution_i_num_vars = len(gold_solution_i_tuples_list)
            solution_i_num_vars = len(solution_i_tuples_list)
            if gold_solution_i_num_vars != solution_i_num_vars:
                grades.addMessage('FAIL: {}'.format(self.path))
                grades.addMessage('\tStudent solution differed from autograder solution')
                grades.addMessage('\tNumber of variables in student solution: {}'.format(
                    solution_i_num_vars))
                grades.addMessage('\tNumber of variables in correct solution: {}'.format(
                    gold_solution_i_num_vars))
                return False

            for gold_solution_var in gold_solution_i_dict:
                if gold_solution_var not in solution_i_dict:
                    grades.addMessage('FAIL: {}'.format(self.path))
                    grades.addMessage('\tStudent solution does not contain the same variables as correct solution')
                    grades.addMessage('\tCorrect solution variable missing in student solution: {}'.format(
                        gold_solution_var))
                    return False

            # Some miscellaneous inequality; return which variables are different between solution and student.
            for key in gold_solution_i_dict:
                if gold_solution_i_dict[key] != solution_i_dict[key]:
                    grades.addMessage('FAIL: {}'.format(self.path))
                    grades.addMessage('\tStudent model does not assign the correct value for variable {}'.format(key))
                    grades.addMessage('\tStudent value for {}: {}'.format(key, solution_i_dict[key]))
                    grades.addMessage('\tCorrect value for {}: {}'.format(key, gold_solution_i_dict[key]))
                    if "WALL" in key:
                        grades.addMessage('\tDouble check that you are loading the map properly.')
                    return False

            if str(solution_i) != str(gold_solution_i):
                grades.addMessage('FAIL: {}'.format(self.path))
                grades.addMessage('\tStudent solution differed from autograder solution for {}'.format(self.soln_labels[i]))
                grades.addMessage('\tStudent solution: {}'.format(solution_i))
                grades.addMessage('\tCorrect solution: {}'.format(gold_solution_i))
                return False

        grades.addMessage('PASS: %s' % self.path)
        return True

    def writeSolution(self, moduleDict, filePath):
        logicPlan = moduleDict['logicPlan']
        # open file and write comments
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        print("Solving problem", self.layoutName)
        print(self.layoutText)

        solution = self.solution(logicPlan)

        print("Problem solved")

        for i, solution_i in enumerate(solution):
            handle.write('{}: "{}"\n'.format(self.soln_labels[i], logicPlan.modelToString(solution_i)))
        handle.close()

    # BEGIN SOLUTION NO PROMPT
    def createPublicVersion(self):
        pass
    # END SOLUTION NO PROMPT


class PositionProblemTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(PositionProblemTest, self).__init__(question, testDict)
        self.layoutText = testDict['layout']
        self.layoutName = testDict['layoutName']

    def solution(self, logicPlan):
        lay = layout.Layout([l.strip() for l in self.layoutText.split('\n')])
        pac = logicAgents.LogicAgent('plp', 'PositionPlanningProblem', logicPlan)
        ghosts = []
        disp = textDisplay.NullGraphics()
        games = next(pacman.runGames(lay, pac, ghosts, disp, 1, False, catchExceptions=True, timeout=300))
        gameState = games[0].state
        return (gameState.isWin(), gameState.getScore(), pac.actions)

    def execute(self, grades, moduleDict, solutionDict):
        logicPlan = moduleDict['logicPlan']
        gold_path = solutionDict['solution_path']
        gold_score = int(solutionDict['solution_score'])

        solution = self.solution(logicPlan)

        if not solution[0] or solution[1] < gold_score:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\tpacman layout:\t\t%s' % self.layoutName)
            if solution[0]:
                result_str = "wins"
            else:
                result_str = "loses"
            grades.addMessage('\tstudent solution result: Pacman %s' % result_str)
            grades.addMessage('\tstudent solution score: %d' % solution[1])
            grades.addMessage('\tstudent solution path: %s' % ' '.join(solution[2]))
            if solution[1] < gold_score:
                grades.addMessage('Optimal solution not found.')
            grades.addMessage('')
            grades.addMessage('\tcorrect solution score: %d' % gold_score)
            grades.addMessage('\tcorrect solution path: %s' % gold_path)
            return False

        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\tpacman layout:\t\t%s' % self.layoutName)
        grades.addMessage('\tsolution score:\t\t%d' % gold_score)
        grades.addMessage('\tsolution path:\t\t%s' % gold_path)
        return True

    def writeSolution(self, moduleDict, filePath):
        logicPlan = moduleDict['logicPlan']
        # open file and write comments
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        print("Solving problem", self.layoutName)
        print(self.layoutText)

        solution = self.solution(logicPlan)

        print("Problem solved")

        handle.write('solution_win: "%s"\n' % str(solution[0]))
        handle.write('solution_score: "%d"\n' % solution[1])
        handle.write('solution_path: "%s"\n' % ' '.join(solution[2]))
        handle.close()

    # BEGIN SOLUTION NO PROMPT
    def createPublicVersion(self):
        pass
    # END SOLUTION NO PROMPT


class FoodProblemTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(FoodProblemTest, self).__init__(question, testDict)
        self.layoutText = testDict['layout']
        self.layoutName = testDict['layoutName']

    def solution(self, logicPlan):
        lay = layout.Layout([l.strip() for l in self.layoutText.split('\n')])
        pac = logicAgents.LogicAgent('flp', 'FoodPlanningProblem', logicPlan)
        ghosts = []
        disp = textDisplay.NullGraphics()
        games = next(pacman.runGames(lay, pac, ghosts, disp, 1, False, catchExceptions=True, timeout=300))
        gameState = games[0].state
        return (gameState.isWin(), gameState.getScore(), pac.actions)

    def execute(self, grades, moduleDict, solutionDict):
        logicPlan = moduleDict['logicPlan']
        gold_path = solutionDict['solution_path']
        gold_score = int(solutionDict['solution_score'])

        solution = self.solution(logicPlan)

        if not solution[0] or solution[1] < gold_score:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\tpacman layout:\t\t%s' % self.layoutName)
            if solution[0]:
                result_str = "wins"
            else:
                result_str = "loses"
            grades.addMessage('\tstudent solution result: Pacman %s' % result_str)
            grades.addMessage('\tstudent solution score: %d' % solution[1])
            grades.addMessage('\tstudent solution path: %s' % ' '.join(solution[2]))
            if solution[1] < gold_score:
                grades.addMessage('Optimal solution not found.')
            grades.addMessage('')
            grades.addMessage('\tcorrect solution score: %d' % gold_score)
            grades.addMessage('\tcorrect solution path: %s' % gold_path)
            return False

        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\tpacman layout:\t\t%s' % self.layoutName)
        grades.addMessage('\tsolution score:\t\t%d' % gold_score)
        grades.addMessage('\tsolution path:\t\t%s' % gold_path)
        return True

    def writeSolution(self, moduleDict, filePath):
        logicPlan = moduleDict['logicPlan']
        # open file and write comments
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        print("Solving problem", self.layoutName)
        print(self.layoutText)

        solution = self.solution(logicPlan)

        print("Problem solved")

        handle.write('solution_win: "%s"\n' % str(solution[0]))
        handle.write('solution_score: "%d"\n' % solution[1])
        handle.write('solution_path: "%s"\n' % ' '.join(solution[2]))
        handle.close()

    # BEGIN SOLUTION NO PROMPT
    def createPublicVersion(self):
        pass
    # END SOLUTION NO PROMPT


class LocalizationProblemTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(LocalizationProblemTest, self).__init__(question, testDict)
        self.layoutText = testDict['layout']
        self.layoutName = testDict['layoutName']
        self.scriptedActions = eval(testDict['actions'])

    def solution(self, logicPlan):
        lay = layout.Layout([l.strip() for l in self.layoutText.split('\n')])
        ghosts = []
        # TODO: Figure out if we can use no-graphics cleaner
        disp = self.question.display
        if isinstance(disp, graphicsDisplay.PacmanGraphics): # autograder.py has incorrect options
            disp = graphicsDisplay.PacmanGraphics(frameTime=0.5)
        pac = logicAgents.LocalizationLogicAgent(
            'loc', 'LocalizationProblem', logicPlan, display=disp, scripted_actions=self.scriptedActions)
        yield from pacman.runGames(lay, pac, ghosts, disp, 1, False, catchExceptions=True, timeout=300)

    def execute(self, grades, moduleDict, solutionDict):
        logicPlan = moduleDict['logicPlan']
        gold_solution = eval(solutionDict['possible_locations_per_timestep'])

        num_timesteps = 0
        for t, solution in enumerate(self.solution(logicPlan)):
            if solution is None:
                num_timesteps = t
                break
            if set(solution) != set(gold_solution[t]):
                grades.addMessage('FAIL: {}'.format(self.path))
                grades.addMessage('\tStudent solution differed from autograder solution at timestep t = {}'.format(t))
                grades.addMessage('\tStudent solution at time t = {}: {}'.format(t, solution))
                grades.addMessage('\tCorrect solution at time t = {}: {}'.format(t, gold_solution[t]))
                return False
        
        if num_timesteps != len(gold_solution):
            grades.addMessage('FAIL: {}'.format(self.path))
            grades.addMessage('\tStudent solution differed from autograder solution')
            grades.addMessage('\tStudent solution timestep number: {}'.format(num_timesteps))
            grades.addMessage('\tCorrect solution timestep number: {}'.format(len(eval(solutionDict['possible_locations_per_timestep']))))
            return False

        grades.addMessage('PASS: %s' % self.path)
        return True

    def writeSolution(self, moduleDict, filePath):
        logicPlan = moduleDict['logicPlan']
        # open file and write comments
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        print("Solving problem", self.layoutName)
        print(self.layoutText)

        solution = self.solution(logicPlan)

        print("Problem solved")

        handle.write('possible_locations_per_timestep: "{}"\n'.format(str(solution)))
        handle.close()

    # BEGIN SOLUTION NO PROMPT
    def createPublicVersion(self):
        pass
    # END SOLUTION NO PROMPT

class MappingProblemTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(MappingProblemTest, self).__init__(question, testDict)
        self.layoutText = testDict['layout']
        self.layoutName = testDict['layoutName']
        self.scriptedActions = eval(testDict['actions'])
        self.solution_label = 'known_map_per_timestep'

    def solution(self, logicPlan):
        lay = layout.Layout([l.strip() for l in self.layoutText.split('\n')])
        ghosts = []
        # TODO: Figure out if we can use no-graphics cleaner
        disp = self.question.display
        if isinstance(disp, graphicsDisplay.PacmanGraphics): # autograder.py has incorrect options
            disp = graphicsDisplay.PacmanGraphics(frameTime=0.5, render_walls_beforehand=False)
        pac = logicAgents.MappingLogicAgent(
            'mp', 'MappingProblem', logicPlan, display=disp, scripted_actions=self.scriptedActions)
        yield from pacman.runGames(lay, pac, ghosts, disp, 1, False, catchExceptions=True, timeout=300)

    def check_len(self, grades, soln, gold_soln, str_info=""):
        if len(soln) != len(gold_soln):
            grades.addMessage('FAIL: {}'.format(self.path))
            grades.addMessage('\tstudent solution length {}: {}'.format(str_info, len(soln)))
            grades.addMessage('\tcorrect solution length {}: {}'.format(str_info, len(gold_soln)))
            return False
        return True

    def execute(self, grades, moduleDict, solutionDict):
        logicPlan = moduleDict['logicPlan']
        gold_solution = eval(solutionDict[self.solution_label])

        num_timesteps = 0

        for t, solution_t in enumerate(self.solution(logicPlan)):
            if solution_t == None:
                num_timesteps = t
                break
            if not self.check_len(grades, solution_t, gold_solution[t], "at time t = {}".format(t)):
                return False

            if solution_t != gold_solution[t]:
                grades.addMessage('FAIL: {}'.format(self.path))
                grades.addMessage('\tStudent solution differed from autograder solution at timestep t = {}'.format(t))
                grades.addMessage('\tStudent solution at time t = {}: {}'.format(t, solution_t))
                grades.addMessage('\tCorrect solution at time t = {}: {}'.format(t, gold_solution[t]))
                return False
        
        if num_timesteps != len(gold_solution):
            grades.addMessage('FAIL: {}'.format(self.path))
            grades.addMessage('\tStudent solution differed from autograder solution')
            grades.addMessage('\tStudent solution timestep number: {}'.format(num_timesteps))
            grades.addMessage('\tCorrect solution timestep number: {}'.format(len(eval(solutionDict[self.solution_label]))))
            return False

        grades.addMessage('PASS: %s' % self.path)
        return True

    def writeSolution(self, moduleDict, filePath):
        logicPlan = moduleDict['logicPlan']
        # open file and write comments
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        print("Solving problem", self.layoutName)
        print(self.layoutText)

        solution = self.solution(logicPlan)

        print("Problem solved")

        handle.write('{}: "{}"\n'.format(self.solution_label, str(solution)))
        handle.close()

    # BEGIN SOLUTION NO PROMPT
    def createPublicVersion(self):
        pass
    # END SOLUTION NO PROMPT

class SLAMProblemTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(SLAMProblemTest, self).__init__(question, testDict)
        self.layoutText = testDict['layout']
        self.layoutName = testDict['layoutName']
        self.scriptedActions = eval(testDict['actions'])
        self.solution_labels = ['known_map_per_timestep', 'possible_locations_per_timestep']

    def solution(self, logicPlan):
        lay = layout.Layout([l.strip() for l in self.layoutText.split('\n')])
        ghosts = []
        # TODO: Figure out if we can use no-graphics cleaner
        disp = self.question.display
        if isinstance(disp, graphicsDisplay.PacmanGraphics): # autograder.py has incorrect options
            disp = graphicsDisplay.PacmanGraphics(frameTime=0.5, render_walls_beforehand=False)
        pac = logicAgents.SLAMLogicAgent(
            'slam', 'SLAMProblem', logicPlan, display=disp, scripted_actions=self.scriptedActions)
        yield from pacman.runGames(lay, pac, ghosts, disp, 1, False, catchExceptions=True, timeout=1800)

    def check_len(self, grades, soln, gold_soln, str_info=""):
        if len(soln) != len(gold_soln):
            grades.addMessage('FAIL: {}'.format(self.path))
            grades.addMessage('\tstudent solution length {}: {}'.format(str_info, len(soln)))
            grades.addMessage('\tcorrect solution length {}: {}'.format(str_info, len(gold_soln)))
            return False
        return True

    def execute(self, grades, moduleDict, solutionDict):
        logicPlan = moduleDict['logicPlan']
        num_timesteps = 0
        for t, solutions_at_t in enumerate(self.solution(logicPlan)):
            if solutions_at_t is None:
                num_timesteps = t
                break
            for soln_label, solution in zip(self.solution_labels, solutions_at_t):
                gold_solution = eval(solutionDict[soln_label])

                if solution != gold_solution[t]:
                    grades.addMessage('FAIL: {}'.format(self.path))
                    grades.addMessage('\tStudent solution differed from autograder solution at timestep t = {}'.format(t))
                    grades.addMessage('\tStudent solution for {} at time t = {}: {}'.format(soln_label, t, solution))
                    grades.addMessage('\tCorrect solution for {} at time t = {}: {}'.format(soln_label, t, gold_solution[t]))
                    return False
        
        if num_timesteps != len(eval(solutionDict[self.solution_labels[0]])):
            grades.addMessage('FAIL: {}'.format(self.path))
            grades.addMessage('\tStudent solution differed from autograder solution')
            grades.addMessage('\tStudent solution timestep number: {}'.format(num_timesteps))
            grades.addMessage('\tCorrect solution timestep number: {}'.format(len(eval(solutionDict[self.solution_labels[0]]))))
            return False

        grades.addMessage('PASS: %s' % self.path)
        return True

    def writeSolution(self, moduleDict, filePath):
        logicPlan = moduleDict['logicPlan']
        # open file and write comments
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        print("Solving problem", self.layoutName)
        print(self.layoutText)

        solution = self.solution(logicPlan)

        print("Problem solved")

        for soln_label, solution_i in zip(self.solution_labels, solution):
            handle.write('{}: "{}"\n'.format(soln_label, str(solution_i)))
        handle.close()

    # BEGIN SOLUTION NO PROMPT
    def createPublicVersion(self):
        pass
    # END SOLUTION NO PROMPT

class LogicStatementTest(testClasses.TestCase):
 
    def __init__(self, question, testDict):
        super(LogicStatementTest, self).__init__(question, testDict)
        self.preamble = compile(testDict.get('preamble', ""), "%s.preamble" % self.getPath(), 'exec')
        self.test = compile(testDict['test'], "%s.test" % self.getPath(), 'eval')
        self.pairs = testDict['pairs']
        self.success = testDict['success']
        self.failure = testDict['failure']
 
    def evalCode(self, moduleDict):
        bindings = dict(moduleDict)
        exec(self.preamble, bindings)
        return eval(self.test, bindings)
 
    def execute(self, grades, moduleDict, solutionDict):
        bindings = dict(moduleDict)
        exec(self.preamble, bindings)
        truths = eval(self.test, bindings)
        model_truth_pairs = eval(self.pairs, bindings)
        if str(truths) == solutionDict['result']:
            grades.addMessage('PASS: %s' % self.path)
            grades.addMessage('\t%s' % self.success)
            return True
        else:
            solution_truths = eval(solutionDict['result'])
            firstError = 1
            while truths[firstError-1] == solution_truths[firstError-1]:
                firstError += 1
            model = model_truth_pairs[firstError-1][0]
 
            grades.addMessage('FAIL: %s' % self.path)
            # grades.addMessage('\t%s' % self.failure)            
            grades.addMessage('Your solution\'s first error occurred on model %d.' % firstError)
            grades.addMessage('MODEL:  %s' % model)
            grades.addMessage('The correct answer is %s but you returned %s.' % (solution_truths[firstError-1], truths[firstError-1]))
 
 
        return False
 
    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# The result of evaluating the test must equal the below when cast to a string.\n')
 
        handle.write('result: "%s"\n' % self.evalCode(moduleDict))
        handle.close()
        return True
 
    # BEGIN SOLUTION NO PROMPT
    def createPublicVersion(self):
        pass
    # END SOLUTION NO PROMPT

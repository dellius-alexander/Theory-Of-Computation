from itertools import combinations
from numbers import Number
from re import S, T
import string
from datascience.util import make_array
from sympy import symbols, init_printing, pretty_print, Integral, \
    sin, cos, tan, cot, sec, csc, pretty, Range
from sympy.core import symbol
from math import ceil
from __gcd import GCD
from datascience import Table
import numpy as np
import plotly.graph_objects as go
from pandas import DataFrame 


init_printing(printer='true')


class proposition(object):
    """
    Mathematical logic includes predicate logic and propositional logic. Each of the following two sentences is a proposition, that is, a description of a specific domain:

    (1) The earth goes around the sun.
    (2) A telephone is a terminal for telecommunication services.

    Each proposition is composed of a primitive proposition or a compound proposition, which is a combination of primitive propositions.

    Every proposition is either true (T) or false (F). Proposition (1) and (2) are true. On the other hand, the propositions “The earth is a triangle” and “A human is a bird” are false.
    10.3.1 Definition of a Proposition and Its Operations

    A proposition can be represented by a symbol, called an atomic formula, such as p, q, r, and s. An atomic formula is either true (T) or false (F).

    Logic involves propositions, which have truth values, either the value true or the value false. The propositions “0 = 1” and “peanut butter is a source of protein” have truth values false and true, respectively. When a simple proposition, which has no variables and is not constructed from other simpler propositions, is used in a logical argument, its truth value is the only information that is relevant.
 
    An operator that performs an operation on one or more atomic formulas is called a logical symbol, for example, ↔, ~, →, ∨, or ∧ The priority of operators is as follows:

    (1) ~
    (2) ∧
    (3) ∨
    (4) →
    (5) ↔

    where (1) is the highest priority and (5) is the lowest priority. The operators have the following meanings:

        “~”,"" is a negation or NOT

        “∧” is a conjunction or AND

        “∨” is a disjunction or OR

        “→” is an implication or "if ... then"

        “↔” is an equivalent/biconditional or "if and only if"


    (1) Both an atomic formula and a negation of an atomic formula are logical expressions.
    (2) A formula of logical expressions connected by an operator or operators is a logical expression.
    (3) Only a formula defined as in items (1) and (2) is a logical expression.

    A proposition involving a variable (a free variable, terminology we will explain shortly) may be true or false, depending on the value of the variable. If the domain, or set of possible values, is taken to be N , the set of non-negative integers, the proposition “x − 1 is prime” is true for the value x = 8 and false when x = 10.

    Compound propositions are constructed from simpler ones using logical connectives. In each case, p and q are assumed to be propositions.

    Each of these connectives is defined by saying, for each possible combination of truth values of the propositions to which it is applied, what the truth value of the result is. The truth value of ¬p is the opposite of the truth value of p.

    The proposition p ∧ q (“p and q”) is true when both p and q are true and false in every other case. “p
    or q” is true if either or both of the two propositions p and q are true, and false only when they are both false.

    The conditional proposition p → q, “if p then q”, is defined to be false when p is true and q is false; one way to understand why it is defined to be true in the
    other cases is to consider a proposition like 

    - x< 1 → x< 2

    where the domain associated with the variable x is the set of natural numbers. It sounds reasonable to say that this proposition ought to be true, no matter what value is substituted for x, and you can see that there is no value of x that makes x < 1 true and x < 2 false. When x = 0, both x < 1 and x < 2 are true; when x = 1, x < 1 is false and x < 2 is true; and when x = 2, both x < 1 and x < 2 are false; therefore, the truth table we have drawn is the only possible one if we want this compound proposition to be true in every case.

    In English, the word order in a conditional statement can be changed without changing the meaning. The proposition

    - p → q

    can be read either:
    “if p then q” or “q if p”. In both cases, the “if ” comes right before p
    “p only if q”, may seem confusing until you realize that “only if ” and “if ” mean different things.
    The English translation of the bi-conditional statement

    - p ↔ q

    is a combination of:
    “p if q” and “p only if q”. The statement is true when the truth values of p and q are the same and false when they are different.

    Once we have the truth tables for the five connectives, finding the truth values for an arbitrary compound proposition constructed using the five is a straightforward operation. We illustrate the process for the proposition

    - (p ∨ q) ∧ ¬(p → q)

    We begin filling in the table below by entering the values for p and q in the two leftmost columns; if we wished, we could copy one of these columns for each occurrence of p or q in the expression. The order in which the remaining columns are filled in (shown at the top of the table) corresponds to the order in which the operations are carried out, which is determined to some extent by the way the
    expression is parenthesized.

    The first two columns to be computed are those corresponding to the sub-expressions p ∨ q and p → q. Column 3 is obtained by negating column 2, and the final result in column 4 is obtained by combining columns 1 and 3 using the ∧ operation.

    A tautology is a compound proposition that is true for every possible combination of truth values of its constituent propositions—in other words, true in every case.

    According to the definition of the bi-conditional connective, p ↔ q is true precisely when p and q have the same truth values. One type of tautology, therefore, is a proposition of the form P ↔ Q, where P and Q are compound propositions that are logically equivalent—i.e., have the same truth value in every possible case. Every proposition appearing in a formula can be replaced by any other logically equivalent proposition, because the truth value of the entire formula remains unchanged. We write

    - P ⇔ Q

    to mean that the compound propositions P and Q are logically equivalent.
    A related idea is logical implication. We write

    - P ⇒ Q 

    to mean that in every case where P is true, Q is also true, and we describe this situation by saying that P logically implies Q. 
  
    The operation rules governing logical expressions are as follows:

    (1) p ↔ q = (p → q) ∧ (p → p)
    (2) p → q = ~ p ∨ q
    (3) p ∧ T = p
    (4) p ∧ F = F
    (5) p ∨ T = T
    (6) p ∨ F = p
    (7) ~ (~ p) = p
    (8) p ∧ p = p
    (9) p ∨ p = p
    (10) \p ∨ ~ p = T
    (11) p ∧ ~ p = F
    (12) p∧ q = q∧ p
    (13) p ∨ q = q ∨ p
    (14) p ∧∧ (q ∨ r) = (p ∧ q) ∨ (p ∧r)
    (15) p ∨ (q ∧ r) = (p ∨ q) ∧ (p ∨ r)
    (16) ~(p ∧ q) = ~ p ∨ ~ q
    (17) ~(p ∨ q)= ~ p ∧ ~ q
    (18) (p ∧ q) ∧ r = p ∧ (q ∧ r)
    (19) (p ∨ q) ∨ r = p ∨ (q∨ r)
    # https://www.sciencedirect.com/topics/computer-science/prepositional-logic
    # https://en.wikipedia.org/wiki/Propositional_calculus
    
    """

    def __init__(self, language=None or string,listOfElems=None):
        """
        
        :param language: a string of propositional argumants and operators
        :param listOfElems: a list containing all letters of the proposition
        """
        self.language = language
        self.operators = {
            'negation': ['~','¬'],
            'conjunction': ['^', '∧'],
            'disjunction': ['v'],
            'implication': ['→', '->', '-->'],
            'equivalent': ['↔', '<->', '<-->'],
        }
        self.listOfElems = listOfElems
        self.letters = [] if self.language is None else [a for a in self.language if a != ' ' and a != 'v']
        self.table = []
        org_cap_alpha = str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-=()")
        sup_cap_alpha = str("⁰¹²³⁴⁵⁶⁷⁸⁹ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᵠᴿˢᵀᵁⱽᵂˣʸᶻ⁺⁻⁼⁽⁾")
        sub_cap_alpha = str("₀₁₂₃₄₅₆₇₈₉ₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧZ₊₋₌₍₎")
        self.super_trans = str.maketrans(org_cap_alpha, sup_cap_alpha)
        self.subs_trans = str.maketrans(org_cap_alpha, sub_cap_alpha)


    #####################################################################
    def checkIfDuplicates_2(self,listOfElems=None):
        ''' Check if given list contains any duplicates 
        :param listOfElems: a list containing all letters of the proposition
        '''    
        if listOfElems is None:
            listOfElems = self.listOfElems
        setOfElems = set()
        for elem in listOfElems:
            if elem in setOfElems:
                return True
            else:
                setOfElems.add(elem)         
        return False

    #####################################################################
    def isEven(self, num=Number) -> object:
        """
        :param num: The number being evaluated
        :return: Return T if number is even or F if number is odd.
        """
        num = int(num)
        if (num % 2) == 0:
            # print("{0} is Even".format(num))
            return 'T'
        else:
            # print("{0} is Odd".format(num))
            return 'F'

 #####################################################################
    def __get_indexes(self,proposition_list=list, data_dictionary=list) -> list:
        """
        Gets the indexes of each recurring proposition
        
        :param proposition_list: a list of prepositional arguments; i.e. ['p','q']
        :param data_dictionary: a dictionary of prepositional arguments and their attributes
        """
        p_l = proposition_list
        dict_index = 0
        prev_obj = None
        for rcd in data_dictionary:
            prop_index = 0
            temp = []
            for obj in proposition_list:
                temp.append(obj)                
                if rcd['symbol'] == obj:
                    data_dictionary[dict_index]['indexes'].append(prop_index)
                if prev_obj == "(" and temp[len(temp)-1] == obj or prev_obj == ")" and temp[len(temp)-2].isalpha():
                    data_dictionary[dict_index]['parenthesis'] = True
                prop_index += 1
                prev_obj = obj
            dict_index += 1
        return data_dictionary         

    #####################################################################
    def get_data_dict(self, letters=None or list) -> object:
        """
        creates a map of variables and occurrances

        :param letters: is a list of alphabet from the proposition
        """
        letters = self.letters if letters is None else letters
        get_sym_data = lambda symbol,count,repeated=bool,parenthesis=bool:  {'symbol':symbol,'count': count,'repeated': repeated, 'indexes': [], 'parenthesis': parenthesis}
        sym_data = []
        for a in letters: # check for duplicate letters
            found = False
            exists = False
            for b in letters:
                if a == b:
                    found = True
            cntr = 0
            # check if symbol already exist in sym_data
            for k in sym_data:
                # print('{}'.format(k['symbol']))             
                if k['symbol'] == a:
                    exists = True
                    k['repeated'] = True
                    k['count'] += 1
                    sym_data[cntr] = k
                    break
                elif k['symbol'] == None:
                    sym_data.remove(k)
                cntr += 1
            if found == True and exists == False:
                sym_data.append(get_sym_data(a,1,False,False))
        return sym_data
    
    #####################################################################
    def get_truth_table(self,rows=int,cols=int,col_index=None,arr=None) -> list:
        """
        number sequence:
        - Use the formula a_n = a_1 + d(n − 1) to identify the arithmetic sequence.

        - Sequence: 0,2,4,6 = a_n = 2n - 2, formula for even sequence

        - Sequence: 1,3,5,7 = a_n = 2n - 1, formula for odd sequence

        - Sequence: 1,2,4,8 = a_n = 2^n-1, formula for outer loop sequence

        - Sequence: 8,4,2,1 = a_n = N / (2^n-1), formula for inner loop sequence
        
        2 Columns
        ---------
        | p | q |
        --- | ---
        | T | T | <==
        | F | T | <== 4 Rows
        | T | F | <==
        | F | F | <==
        ---   ---
        ***********************

        :param rows: table row count
        :param cols: table column count
        :param col_index: [optional] column index currently being built
        :param arr: [optional] empty array 
        """
        # print('rows: {}; cols: {}; Column Index: {}; Input Array: {}'.format(int(rows), int(cols), col_index, arr))
        tbl = arr if arr is not None else []*cols
        tmp_arr = []
        cnt = col_index if col_index is not None else 1
        seq = int(2**(cnt - 1)) # loop sequence count
        #####################################################################
        # Test case:
        # rows: 4, cols: 2, chg_rate: ceil(rows**cnt)/2 = 2, for current recursive call
        chg_rate = int(ceil((rows/2)/(2**(cnt-1))))
        tmp_arr.clear()
        # used seq to control outer loop interations
        for r in Range(0,seq):      
            # print('Itteration(r): {}; cnt: {}; seq: {}'.format(r,cnt,seq))
            # used chg_rate to control inner loop counter
            for i in Range(0,chg_rate): # Even/True loop
                i = i + 1
                # print('\n- 1 - Column: {} | Row: {} | I: {} | T/F: {}'.format((cols),r,i,((2*i)-2))
                tmp_arr.append(self.isEven((2*i)-2))
            # used chg_rate to control inner loop counter
            for i in Range(0,chg_rate): # Odd/False loop
                i = i + 1
                # print('\n- 1 - Column: {} | Row: {} | I: {} | T/F: {}'.format((cols),r,i,((2*i)-1)))
                tmp_arr.append(self.isEven((2*i)-1))  
        # print('\nColumn: {}\nRow: {}'.format(cols,rows))
        tbl.append(tmp_arr)
        # tbl.reverse()
        # print('Count: {}'.format(cnt))   
        if cnt == cols:
            tbl.reverse()
            return tbl
        return self.get_truth_table(rows,cols,(cnt+1), tbl)
        
    #####################################################################
    def print_array(self, arr=list):
        """
        Prints each row of array line by line.
        :param arr: the input array.
        """
        row_cnt = len(arr)
        for a in Range(row_cnt):
            print('{} \n'.format(arr[a]))
            row_cnt -= 1
        
    #####################################################################
    def get_proposition_tbl(self,language=None,operators=None) -> list:
        """
        Gets the dictionary object containing all propositional arguments, operators and their truth tables

        :param language: a string of propositional argumants and operators
        :return list:
        """
        language = [a for a in self.language if a != ' '] if language is None else [a for a in language if a != ' ']
        operators = self.operators if operators is None else operators
        # initialize first data object
        data_dict = [] 
        # print(len(operators))
        # collect all the alphabets and operators from the proposition
        prop_obj_array = [c for c in language if c != ' ' and c != 'v']  
        prop_operator_array = [a for a in language if a.isalpha() == False and a != ' ' or a == 'v']
        # # print('Propositional objects: {}\nOperators: {}\n'.format(prop_obj_array,prop_operator_array))
        # # print("Proposition/Language: {}, {}".format(len(language),language))
        # # print("Letters: {}, {}".format(len(prop_obj_array),prop_obj_array))
        # # print("Operators/Symbols: {}, {}\n\n".format(len(prop_operator_array),prop_operator_array))
        # first map variables in data dictionary
        data_dict = self.get_data_dict(prop_obj_array)
        # print('{}\n'.format(data_dict))
        # Next get the index of each occurance of each variable
        data_dict = self.__get_indexes(prop_obj_array,data_dict)
        # # print('Data dictionary: \n\t{}\n'.format(data_dict))
        cols = len(data_dict)
        rows = 2**cols
        print('rows: {}; cols: {}\n'.format(rows, cols))
        # Initialize the table for desired rows and columns
        # __table = [[]*rows]*cols
        __table = []
        # Get the truth table for each column
        tbl_cols = self.get_truth_table(rows,cols)
        # add the truth table to the data dictionary
        tbl_cols_len = len(tbl_cols)
        for a in range(tbl_cols_len):
            data_dict[a]['Truth Table'] = tbl_cols[a]
            # print(data_dict[a])

        # Merge the data dictionary and table columns into one table
        for col in range(cols):
            __table.append({data_dict[col]['symbol']:tbl_cols[col]})
        # self.print_array(__table)        
        df = DataFrame(data_dict)
        self.data_dictionary = __table
        self.table = df
        print(self.data_dictionary)
        print(self.table)


#####################################################################
#                               MAIN                                #
#####################################################################
if __name__ == "__main__":
    # __evaluate_prop("p ∨ q v s ^ k v o")
    lang = "p v q"
    symb = ['p','q','g','h','m']
    prop = proposition(lang)
    prop.get_proposition_tbl()
    # cols_ = len(symb)
    # rows_ = 2**cols_
    # t_table = get_truth_table(rows_,cols_)
    # # # print('\nTruth Table:\n--------------------')
    # print_array(t_table)
    # # print('--------------------\n')
    # tbl_cnt = len(t_table)
    # new_tbl = Table()
    # fig = go.Figure(data=[go.table(
    #     header=dict(values=symb), cells=dict(values=[t_table]))])
    # fig.show()


    # fish_measures = {'Fish': ['Angelfish', 
    #                           'Zebrafish', 
    #                           'Killifish', 
    #                           'Swordtail'],
    #                  'Length':[15.2, 6.5, 9, 6],
    #                  'Width': [7.7, 2.1, 4.5, 2]}

    # zebrafish_index = fish_measures['Fish'].index('Zebrafish')

    # zebrafish_length = fish_measures['Length'][zebrafish_index]
    # print(f"The length of a zebrafish is {zebrafish_length:.2f} cm")
    # tt = {
    #     'p': ['T','F','T','F'],
    #     'q': ['T','T','F','F']
    # }

    # df = df(fish_measures)
    # print(df)
    # booleans = [name in ['Angelfish', 'Swordtail']
    #             for name in df.Fish]
    # print(booleans)
    # print(df[booleans])
    # print(df.p.values)
    # assert type(df.p.values) == np.ndarray
    # for t in t_table:
    #     print('r: {}; '.format(t))
    #     # new_tbl.append_column(str(symb[tbl_cnt]),t)
    # new_tbl.to_csv('truth_tbl.csv')
    # # print(new_tbl)
    # new_tbl.show()
echo "# Theory Of Computation" >> README.md && \
git init && \
git add README.md && \
git commit -m "first commit" && \
git branch -M main && \
git remote add origin https://github.com/dellius-alexander/Theroy of Computation.git && \
git push -u origin main
import math
from numpy import array
from numpy.ma import count
from sympy import factorint, divisors


class GCD(object):
    """

    - Dividend % Divisor = Remainder

    - Dividend/Divisor = Quotient

    - Dividend = Divisor × Quotient
    
    """
    # constructor
    def __init__(self, dividend=None or int, divisor=None or int, power=None or int):
        self.dividend = None
        self.divisor = None
        self.pwr = None
        self.dividend = dividend
        self.divisor = divisor
        self.pwr = power
        # self._gcd = self.gcd() if self.gcd() is not None else None
        # print(f'Line [16] : {self.gcd}')
        # self.pwrs = self.pwrss() if self.pwrss() is not None else None
        # print(f'Line [18] : {self.pwrs}')
        # Define superscript and subscript translations
        org_cap_alpha = str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-=()")
        sup_cap_alpha = str("⁰¹²³⁴⁵⁶⁷⁸⁹ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᵠᴿˢᵀᵁⱽᵂˣʸᶻ⁺⁻⁼⁽⁾")
        sub_cap_alpha = str("₀₁₂₃₄₅₆₇₈₉ₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧZ₊₋₌₍₎")
        self.super_trans = str.maketrans(org_cap_alpha, sup_cap_alpha)
        self.subs_trans = str.maketrans(org_cap_alpha, sub_cap_alpha)

    ###############################################################################
    ###############################################################################
    def gcd(self) -> int:
        """Euclidean Algorithm is used to find the Greatest Common Divisor (GCD).

                The Euclidean Algorithm finds the greatest common divisor between two
                numbers/values by computing the remainder of two values, then afterwards
                we shuffle all values << left, preparing new values for the next iteration.
                This continues until the value of the remainder reaches is zero.
                  a: dividend
                  num: divisor
                  r: remainder
                  [ a % num = r ]
                  ■ Read as: a MOD num = r
                  r = 1   # we initialize r before we enter our loop
                  while (r > 0):
                      r = a % num
                      a = num
                      num = r  as r approaches 0
                  GCD = num when r crosses the bounds of 0
                :rtype: int
                :param self:
                                dividend: The dividend
                                divisor: The divisor
                :return: The GCD of a and num is ...
        """
        if not type(self.dividend) is int:
            raise TypeError(f'Dividend [\t{self.dividend}\t] must be an integer...')
        if not type(self.divisor) is int:
            raise TypeError(f'Divisor [\t{self.divisor}\t] must be an integer...')
        a = self.dividend  # Holds the original value of a
        b = self.divisor  # Holds the original value of num
        r = math.inf  # This our remainder initialized to infinity
        # print(f'\nThe remainders of "{a}" MOD "{num}" are:\n|')
        if b < 0:  # Check for negative values & convert to opposite value
            b = -b
        if a < 0:  # Check for negative values & convert to opposite value
            a = -a
        if b == 0:  # Terminate execution on predicate (num = 0)
            # print(f'|\t{a}')
            # print(f'\nThe GCD of "{aa}" and "{bb}"  is: {a}.')
            return a  # if the divisor is 0 return dividend
        # print(f'|\t{a}')
        # print(f'|\t{num}')
        while r > 0:
            r = a % b
            a = b
            b = r
            # print(f'|\t{num}')
            ans = a
        # print(f"""\nThe GCD of "{aa}" and "{bb}"  is: {ans}.\n""")
        return ans

    ###############################################################################

    def gcd_euclid(self=None, dividend=None, divisor=None) -> int:
        """Euclidean Algorithm is used to find the Greatest Common Divisor (GCD).

                The Euclidean Algorithm finds the greatest common divisor between two
                numbers/values by computing the remainder of two values, then afterwards
                we shuffle all values << left, preparing new values for the next iteration.
                This continues until the value of the remainder reaches is zero.
                  a: dividend
                  num: divisor
                  r: remainder
                  [ a % num = r ]
                  ■ Read as: a MOD num = r
                  r = 1   # we initialize r before we enter our loop
                  while (r > 0):
                      r = a % num
                      a = num
                      num = r  as r approaches 0
                  GCD = num when r crosses the bounds of 0
                :param divisor:
                :param dividend:
                :param self:
                                dividend: The dividend
                                divisor: The divisor
                :return: The GCD of a and num is ...
                """
        if dividend is not None and divisor is not None:
            divid = dividend
            divsr = divisor
        else:
            print(f'|gcd| Exiting......!!!\nCheck your input......!!!\n{self}\n')
            exit(0)
        if not type(divid) is int:
            raise TypeError(f'Dividend [\t{divid}\t] must be an integer...')
        if not type(divsr) is int:
            raise TypeError(f'Divisor [\t{divsr}\t] must be an integer...')
        a = divid  # Holds the original value of a
        b = divsr  # Holds the original value of num
        r = math.inf  # This our remainder initialized to infinity
        # print(f'\nThe remainders of "{a}" MOD "{num}" are:\n|')
        if b < 0:  # Check for negative values & convert to opposite value
            b = -b
        if a < 0:  # Check for negative values & convert to opposite value
            a = -a
        if b == 0:  # Terminate execution on predicate (num = 0)
            # print(f'|\t{a}')
            # print(f'\nThe GCD of "{aa}" and "{bb}"  is: {a}.')
            return a  # if the divisor is 0 return dividend
        # print(f'|\t{a}')
        # print(f'|\t{num}')
        while r > 0:
            r = a % b
            a = b
            b = r
            # print(f'|\t{num}')
            ans = a
        # print(f"""\nThe GCD of "{aa}" and "{bb}"  is: {ans}.\n""")
        return ans

    ###############################################################################

    def clear(self=None) -> None:
        """Delete all values from our
        dividend, divisor and power.
        Checks to ensure null operands upon completion.
        :return: None
        """
        self.dividend = None
        self.divisor = None
        self.pwr = None
        if self.dividend is not None:
            print(f'Something went wrong the dividend container is not empty')
        if self.divisor is not None:
            print(f'Something went wrong the divisor container is not empty')
        if self.pwr is not None:
            print(f'Something went wrong the power container is not empty')
        print(f'\nDividend, divisor, power has been converted to None values successfully!!!\n')

    ###############################################################################

    def euclid(self=None, dividend=None, divisor=None) -> int:
        """Euclidean Algorithm is used to find the Greatest Common Divisor (GCD).

        The Euclidean Algorithm finds the greatest common divisor between two
        numbers/values by computing the remainder of two values, then afterwards
        we shuffle all values << left, preparing new values for the next iteration.
        This continues until the value of the remainder reaches is zero.
          a: dividend
          num: divisor
          r: remainder
          [ a % num = r ]
          ■ Read as: a MOD num = r
          r = 1   # we initialize r before we enter our loop
          while (r > 0):
              r = a % num
              a = num
              num = r  as r approaches 0
          GCD = num when r crosses the bounds of 0
        :param divisor:
        :param dividend:
        :param self:
                        dividend: The dividend
                        divisor: The divisor
        :return: The GCD of a and num is ...
        :rtype: int
        """
        if dividend is not None and divisor is not None:
            divid = dividend
            divsr = divisor
        else:
            divid = self.dividend
            divsr = self.divisor
        if not type(divid) is int:
            raise TypeError(f'Dividend [\t{self.divid}\t] must be an integer...')
        if not type(divsr) is int:
            raise TypeError(f'Divisor [\t{self.divsr}\t] must be an integer...')
        a = divid  # Holds the original value of a
        b = divsr  # Holds the original value of num
        r = math.inf  # This our remainder initialized to infinity
        # print(f'\nThe remainders of "{a}" MOD "{num}" are:\n|')
        if b < 0:  # Check for negative values & convert to opposite value
            b = -b
        if a < 0:  # Check for negative values & convert to opposite value
            a = -a
        if b == 0:  # Terminate execution on predicate (num = 0)
            # print(f'|\t{a}')
            # print(f'\nThe GCD of "{aa}" and "{bb}"  is: {a}.')
            return a
        print(f'---------------------------------')
        print(f'\na = {a}')
        print(f'b = {b}')
        print(f'r = a % b')
        while r > 0:
            r = a % b
            print(f'---------------------------------')
            print(f'r = {a} % {b} = {r}')
            print(f'a = {b}')
            print(f'b = {r}')
            a = b
            b = r
            ans = r
        # print(f"""\nThe GCD of "{aa}" and "{bb}"  is: {ans}.\n""")
        print(f'---------------------------------')
        return ans

    ###############################################################################

    def knuth(self=None, dividend=None, divisor=None) -> list:
        """Knuth GCD algorithm computes the gcd(a,num) and finds the pairs of
        integers (x,y) that when applied to Ax + By = z or Ax + By -z = 0.

        Let A and B be  fixed integers, and consider the equation:

        Ax + By = z; eqivalently, Ax + By - z = 0.
        In the three variables (x, y, z), suppose we have two solutions
        (x1, y1, z1) and (x2, y2, z2).  Consider that z1 = A is your
        dividend and z2 = B your divisor in the equation Ax + By = z,
        then for any value of A/B = q (quotient), we can construct another
        solution: (x1, y1,z1) - q(x2, y2, z2) = (x1 - qx2, y1 - qy2, z1 - qz2).
        To  verify that  if  both Ax1 + By1 = z1 and Ax2 + By2 = z2, then
        it is also true that A(x1 - qx2) + B(y1 - qy2) = (z1 - qz2).

        Therefore:
          Ax + By = z
          A = z1
          B = z2
          [x1, y1, z1] = z1
          [x2, y2, z2] = z2
          z1/z2 = q

          [x1, y1, z1] - q[x2, y2, z2] = [x3, y3, z3]
          z2/z3 = q
          [x2, y2, z2] - q[x3, y3, z3] = [x4, y4, z4]
          ...
          iterate and update each set until z = 0 to find our GCD set.

          This example illustrates how the algorithm solves this equation
          by finding the GCD as well as all solutions of (x,y) for z, if
          and only if z is a multiple of the gcd(a,num):

            39x + 17y = z

            [1, 0, 39] = 39  # (1, 0) are true values for (x, y)
            [0, 1, 17] = 17  # (0, 1) are true values for (x, y)

            39/17 = 2.294
            [1, 0, 39] - 2[0, 1, 17] = []
            [1, 0, 39] + [0, -2, -34] = [1, -2, 5]

            17/5 = 3.4
            [0, 1, 17] - 3[1, -2, 5] = []
            [0, 1, 17] + [-3, 6, -15] = [-3, 7, 2]

            5/2 = 2.5
            [1, -2, 5] - 2[-3, 7, 2] = []
            [1, -2, 5] + [6, -14, -4] = [7, -16, 1]

            2/1 = 2
            [-3, 7, 2] - 2[7, -16, 1] = []
            [-3, 7, 2] + [-14, 32, -2] = [-17, 39, 0]

            GCD = [7, -16, 1]
        :param dividend: dividend: The dividend
        :param divisor: divisor: The divisor
        :param self:
        :return: The GCD solution set...
        :rtype: array
        """
        if dividend is not None and divisor is not None:
            divid = dividend
            divsr = divisor
        else:
            divid = self.dividend
            divsr = self.divisor
            if divid < 0:  # Check for negative values & convert to opposite value
                divid = self.dividend = -divid
            if divsr < 0:  # Check for negative values & convert to opposite value
                divsr = self.divisor = -divsr
        if not type(divid) is int:
            raise TypeError(f'Dividend [\t{divid}\t] must be an integer...')
        if not type(divsr) is int:
            raise TypeError(f'Divisor [\t{divsr}\t] must be an integer...')
        if divid < 0:  # Check for negative values & convert to opposite value
            divid = -divid
        if divsr < 0:  # Check for negative values & convert to opposite value
            divsr = -divsr
        v1 = array([1, 0, divsr])
        v2 = array([0, 1, divid])
        z = math.inf
        gcd_values = 0
        # print(f'\nSolutions set:\n[  X,  Y,  Z  ]')
        # print(f"""\n{v1}""")
        # print(f"""\n{v2}""")
        while z > 0:
            q = v1[2] // v2[2]
            v3 = v1 - q * v2
            # print(f"""\n{v3}""")
            v1 = v2
            v2 = v3
            z = v3[2]
        return v1

    ###############################################################################

    def knuth_verbose(self=None, dividend=None, divisor=None) -> list:
        """Knuth GCD algorithm computes the gcd(a,num) and finds the pairs of
        integers (x,y) that when applied to Ax + By = z or Ax + By -z = 0.

        Let A and B be  fixed integers, and consider the equation:

        Ax + By = z; eqivalently, Ax + By - z = 0.
        In the three variables (x, y, z), suppose we have two solutions
        (x1, y1, z1) and (x2, y2, z2).  Consider that z1 = A is your
        dividend and z2 = B your divisor in the equation Ax + By = z,
        then for any value of A/B = q (quotient), we can construct another
        solution: (x1, y1,z1) - q(x2, y2, z2) = (x1 - qx2, y1 - qy2, z1 - qz2).
        To  verify that  if  both Ax1 + By1 = z1 and Ax2 + By2 = z2, then
        it is also true that A(x1 - qx2) + B(y1 - qy2) = (z1 - qz2).

        Therefore:
          Ax + By = z
          A = z1
          B = z2
          [x1, y1, z1] = z1
          [x2, y2, z2] = z2
          z1/z2 = q

          [x1, y1, z1] - q[x2, y2, z2] = [x3, y3, z3]
          z2/z3 = q
          [x2, y2, z2] - q[x3, y3, z3] = [x4, y4, z4]
          ...
          iterate and update each set until z = 0 to find our GCD set.

          This example illustrates how the algorithm solves this equation
          by finding the GCD as well as all solutions of (x,y) for z, if
          and only if z is a multiple of the gcd(a,num):

            39x + 17y = z

            [1, 0, 39] = 39  # (1, 0) are true values for (x, y)
            [0, 1, 17] = 17  # (0, 1) are true values for (x, y)

            39/17 = 2.294
            [1, 0, 39] - 2[0, 1, 17] = []
            [1, 0, 39] + [0, -2, -34] = [1, -2, 5]

            17/5 = 3.4
            [0, 1, 17] - 3[1, -2, 5] = []
            [0, 1, 17] + [-3, 6, -15] = [-3, 7, 2]

            5/2 = 2.5
            [1, -2, 5] - 2[-3, 7, 2] = []
            [1, -2, 5] + [6, -14, -4] = [7, -16, 1]

            2/1 = 2
            [-3, 7, 2] - 2[7, -16, 1] = []
            [-3, 7, 2] + [-14, 32, -2] = [-17, 39, 0]

            GCD = [7, -16, 1]
        :param dividend: dividend: The dividend
        :param divisor: divisor: The divisor
        :param self:
        :return: The GCD solution set...
        :rtype: array
        """
        if dividend is not None and divisor is not None:
            divid = dividend
            divsr = divisor
        else:
            divid = self.dividend
            divsr = self.divisor
            if divid < 0:  # Check for negative values & convert to opposite value
                divid = self.dividend = -divid
            if divsr < 0:  # Check for negative values & convert to opposite value
                divsr = self.divisor = -divsr
        if not type(divid) is int:
            raise TypeError(f'Dividend [\t{divid}\t] must be an integer...')
        if not type(divsr) is int:
            raise TypeError(f'Divisor [\t{divsr}\t] must be an integer...')
        if divid < 0:  # Check for negative values & convert to opposite value
            divid = -divid
        if divsr < 0:  # Check for negative values & convert to opposite value
            divsr = -divsr
        v1 = array([1, 0, divsr])
        v2 = array([0, 1, divid])
        z = math.inf
        gcd_values = 0
        x1 = v1[0]
        y1 = v1[1]
        x2 = v2[0]
        y2 = v2[1]
        print(f'\nKnuth Solutions set:\n[  X,  Y,  Z  ]')
        print(f"""\n{v1} = ax + by = ({v1[2]}*{x1} + {v2[2]}*{y1}) = {(v1[2]*x1)+(v2[2]*y1)}""")
        print(f"""\n{v2} = ax + by = ({v1[2]}*{x2} + {v2[2]}*{y2}) = {(v1[2]*x2)+(v2[2]*y2)}""")
        while z > 0:
            q = int(v1[2] // v2[2])
            v3 = v1 - q * v2
            x1 = v1[0]
            y1 = v1[1]
            x2 = v2[0]
            y2 = v2[1]
            print(f"""\nq = {v1[2]} // {v2[2]} = {v1[2] // v2[2]}""")
            print(f'\n(({x1} - {q}*{x2}), ({y1} - {q}*{y2}), ({v1[2]} - {q}*{v2[2]}))'
                  f' = ({x1 - q*x2}, {y1 - q*y2}, {v1[2] - q*v2[2]})')

            v1 = v2
            v2 = v3
            z = v3[2]
        return v1

    ###############################################################################

    def pwrss(self) -> list:
        """This method return the values of 2 decimal conversion of the given number.

                    This method converts the number to binary, converts each ON ["1"]
                    point to its decimal representation, and returns an array of
                    converted ON point.
                        a ≡ r (mod m) | where r is your residue for each iteration
                        of [ a=1, a=2, ..., m - 1 ] where gcd(a, m) == 1
                :param self:    class attribute
                                power: The number used to find residue values
                :return: array of values
                """
        power = self.pwr
        bin_value = bin(power)  # convert our number to binary
        le = len(bin_value)  # get the length of our binary number
        bin_value = bin_value[2:le]  # split the first two digits
        length = len(bin_value)  # get the length of our binary number
        bin_array1 = list()
        for x in bin_value:
            bin_array1.append(int(x))
            length -= 1
        bin_array1.reverse()  # reverse the array before we check values
        exp = 1  # 2**exp;  this is the exponent of our 2's power
        solutions0 = list()
        lt = 0
        for c in bin_array1:
            lt += 1
            if c == 1:
                solutions0.append(exp)  # assign the 2**n value of our bit
            exp += exp  # increment the exponent after each iteration
        bin_array1.reverse()  # reverse back to standard notation
        solutions0.reverse()
        # print(f'\nBinary representation of {self.power} is:\n {bin_array1}\n {solutions0}')
        solutions0.reverse()
        # print(f'\nThe number [ {self.power} ], binary to decimal conversion ON ["1"]'
        #      f'bits are:\nWhere k =  {solutions0}\n')
        return solutions0

    ###############################################################################

    def powers(self, power=None) -> list:
        """This method return the values of 2 decimal conversion of the given number.

            This method converts the number to binary, converts each ON ["1"]
            point to its decimal representation, and returns an array of
            converted ON point.
                a ≡ r (mod m) | where r is your residue for each iteration
                of [ a=1, a=2, ..., m - 1 ] where gcd(a, m) == 1
        :param self:    class attribute
        :param power:                 power: The number used to find residue values
        :return: array of values
        """
        if power is not None:
            p = power
        else:
            p = self.pwr
        bin_array0 = bin(p)  # convert our number to binary
        le = len(bin_array0)  # get the length of our binary number
        bin_array0 = bin_array0[2:le]  # split the first two digits
        length = len(bin_array0)  # get the length of our binary number
        bin_array1 = list()
        for x in bin_array0:
            bin_array1.append(int(x))
            length -= 1
        bin_array1.reverse()  # reverse the array before we check values
        exp = 1  # 2**exp;  this is the exponent of our 2's power
        solutions = list()
        lt = 0
        for c in bin_array1:
            lt += 1
            if c == 1:
                solutions.append(exp)  # assign the 2**n value of our bit
            exp += exp  # increment the exponent after each iteration
        bin_array1.reverse()  # reverse back to standard notation
        solutions.reverse()
        print(f'\nBinary representation of {p} is:\n {bin_array1}\n {solutions}')
        solutions.reverse()
        print(f'\nThe POWER [ {p} ], binary to decimal conversion ON ["1"]'
              f'bits are:\nWhere k =  {solutions}\n')
        return solutions

    ###############################################################################

    ###############################################################################

    def knuth_streamlined(self, dividend=None, divisor=None) -> list:
        """

        :param self:
        :param dividend:
        :param divisor:
        :return:
        """
        if dividend is not None and divisor is not None:
            divid = dividend
            divsr = divisor
        else:
            divid = self.dividend
            divsr = self.divisor
            if divid < 0:  # Check for negative values & convert to opposite value
                divid = self.dividend = -divid
            if divsr < 0:  # Check for negative values & convert to opposite value
                divsr = self.divisor = -divsr
        if not type(divid) is int:
            raise TypeError(f'Dividend [\t{divid}\t] must be an integer...')
        if not type(divsr) is int:
            raise TypeError(f'Divisor [\t{divsr}\t] must be an integer...')
        if divid < 0:  # Check for negative values & convert to opposite value
            divid = -divid
        if divsr < 0:  # Check for negative values & convert to opposite value
            divsr = -divsr
        s1 = array([0, divsr])
        s2 = array([1, divid])
        gcd_values = 0
        print(f'\nSolutions set:\n[ Y, Z ]')
        q = math.inf
        print(f'\n{s1}')
        print(f'\n{s2}')
        while q > 0:
            q = int(s1[1] / s2[1])
            s3 = s1 + (-q * s2)
            print(f"""\n{s3}""")
            s1 = s2
            s2 = s3
            q = s3[1]
            if s3[1] > 0:
                gcd_values = s3
        return gcd_values

        ###############################################################################

    def knuth_solution(self, zz=None, dividend=None, divisor=None) -> None:
        """The knuth_solution function finds the set (x,y) values,
        where x and y are both integers, when applied to Ax + By = z.

        This sample problem below illustrates how to find the (x,y) values:

          Find the solution to  [ 37x + 17y = 7 ], where
          x and y are both integers.

          To find the solution that makes z true, we find the solution set
          for our gcd(a,num) where [x, y, GCD]; z will be the incrementer of
          our x and y values (z*x, z*y) where [ (a(z*x) + num(z*y)) / gcd = z ]
          Or alternatively where [ a*((z*x)/gcd) + num*((z*y)/gcd) = z ]

        :param dividend: The dividend
        :param divisor: The divisor
        :param zz: The incrementer
        :return: None
        """
        if dividend is not None and divisor is not None and zz is not None:
            divid = dividend
            divsr = divisor
            z = zz
        elif zz is not None and self.dividend is not None and self.divisor is not None:
            divid = self.dividend
            divsr = self.divisor
            z = zz
        else:
            divid = self.dividend
            divsr = self.divisor
            if divid < 0:  # Check for negative values & convert to opposite value
                divid = self.dividend = -divid
            if divsr < 0:  # Check for negative values & convert to opposite value
                divsr = self.divisor = -divsr
            z = 1
        if not type(divid) is int:
            raise TypeError(f'Dividend [\t{divid}\t] must be an integer...')
        if not type(divsr) is int:
            raise TypeError(f'Divisor [\t{divsr}\t] must be an integer...')
        if not type(z) is int:
            raise TypeError(f'Z [\t{z}\t] must be an integer...')
        if z < 1:
            raise Exception('Sorry no "z" values below one...')
        if divid < 0:  # Check for negative values & convert to opposite value
            divid = -divid
        if divsr < 0:  # Check for negative values & convert to opposite value
            divsr = -divsr
        gcd = self.knuth(divid, divsr)
        x = gcd[0]
        y = gcd[1]
        g = gcd[2]
        # [a(z*x) + num(z*y)]/gcd = z
        zz = (divsr * ((z * x) / g)) + (divid * ((z * y) / g))

        print(f"""\n\nThe solution to equation:
      {divsr}x + {divid}y = {z}         
      where x and y are both integers is:
      x = {x} and y = {y} where the GCD = {g}
      x = {(z * x) // g} and y = {(z * y) // g} when z = {zz}
      ({divsr}*((z*x)/{g})) + ({divid}*((z*y)/{g})) = {z}
      ({divsr}*{(z * x) // g}) + ({divid}*{(z * y) // g}) = {zz} \n""")

    ###############################################################################

    def congruent(self, dividend=None) -> None:
        """Is it congruent

            2**n ≡ r (mod n)
            r ≡ 2**n (mod n)


        :param dividend: int
        :return: none
        """
        if dividend is not None:
            divid = dividend
        else:
            divid = self.dividend
        a = 1
        divisor = 1
        m = 1
        j = 1
        c = 1
        wf = open('solutions.txt', 'w+')
        rf = open('solutions.txt', 'r')
        wf.write('\nSolutions List:\n')
        print('\nSolutions List: \n')
        while divisor < divid - 1:
            w = int(divid / divisor)
            quotient = divid / divisor
            difference = quotient - w
            m = divid % divisor
            a = quotient * divisor
            af = open('solutions.txt', 'a')
            if m == 1:  # or m == 0
                line3 = f"""|{j}|  Tracker -> K is: {divisor}\n\n|{j}|  Possible multiple: {quotient} x {divisor} = {a}\n"""
                print(line3)
                af.write(line3)
                c = c + 1
            if difference == 0.0:
                line1 = f'|{j}|  {quotient} x {divisor} = {a}\n'
                print(line1)
                af.write(line1)
                c = c + 1
            # if r == 1:
            # line2 = f'|{j}|  Possible multiple: {d} x {k} = {a}\n'
            # print(line2)
            # af.write(line2)
            # c = c + 1
            divisor = divisor + 1
            j = j + 1
        af.write('\nFinished\n')
        wf.close()
        af.close()
        rf.close()
        print('Finished')

    ###############################################################################

    ###############################################################################

    def euler_phi(self=None, number=None, aa=None, count_not_a_multiple=None) -> int:
        """Theorem (Euler):
            if 'm' is a composite number & 'a' is still relatively prime to 'm', meaning 'a' is not a multiple of 'm'
            if (a, n) = 1 then:
            a**φ(n) ≡ 1 (mod n)
            [ φ(n) ] = | num Σ {1, ... , n-1} &  [ (num,n) = 1 ] or  [ num ≡ 1 (mod n) ] |
            the number of numbers in the range [ 1, ... , m-1 ], which are relative prime to
            'm', meaning each number is not a multiple of 'm';

            example: if m=8 & the sum of the numbers relative prime to '8' without going over
            is equal [ 4 Σ { 1,3,5,7 } a4 ≡ 1 (mod 8) ]

            φ(pq) = (p-1) (q-1)

            In secret we pick 2 prime number for the value of p and q, multiply them together to get your public key,
            :param count_not_a_multiple:
            :param number:
            :param aa:
        """
        if number is not None and aa is not None and count_not_a_multiple is not None:
            num = number
            a = aa
            i = aa
            count_not_a_multiple = count_not_a_multiple
            wf_nop = open("not_a_multiple_of_prime.txt", 'a+')
            wf_mop = open("is_a_multiple_of_prime.txt", 'a+')
            wf_nop.write(f'[')
            wf_mop.write(f'[')
            rf_nop = open("not_a_multiple_of_prime.txt", 'r')
            rf_mop = open("is_a_multiple_of_prime.txt", 'r')
        elif number is not None and aa is None and count_not_a_multiple is None:
            num = number
            a = None
            i = 0
            count_not_a_multiple = 0
            count_is_a_multiple = 0
            not_a_multiple = []
            is_a_multiple = []
            wf_nop = open("not_a_multiple_of_prime.txt", 'w+')
            wf_mop = open("is_a_multiple_of_prime.txt", 'w+')
            wf_nop.write(f'[')
            wf_mop.write(f'[')
            rf_nop = open("not_a_multiple_of_prime.txt", 'r')
            rf_mop = open("is_a_multiple_of_prime.txt", 'r')
        else:
            num = self.divisor
            a = None
            i = 0
            count_not_a_multiple = 0
            count_is_a_multiple = 0
            not_a_multiple = []
            is_a_multiple = []
            wf_nop = open("not_a_multiple_of_prime.txt", 'w+')
            wf_mop = open("is_a_multiple_of_prime.txt", 'w+')
            wf_nop.write(f'[')
            wf_mop.write(f'[')
            rf_nop = open("not_a_multiple_of_prime.txt", 'r')
            rf_mop = open("is_a_multiple_of_prime.txt", 'r')
        count_line = 0
        if a is not None:
            print(f'An exception was caught')
            print(f'num = {num}, a = {a}, i = {i}, curr_count = {count_not_a_multiple}')
            print(f'a = {a}')
            try:
                b = self.gcd(a, num)
                # print(f'a = {a}, num = {num}')
                if b == 1:
                    count_not_a_multiple += 1
                    # print(f'a = {a}, num = {num}')
                    # not_a_multiple.append(num)
                    wf_nop.write(f'{a}')
                    # print(f'Current test value: {a}')
                if b != 1:
                    count_is_a_multiple += 1
                    # print(f'a = {a}, num = {num}')
                    # is_a_multiple.append(a)
                    wf_mop.write(f'{a}, ')
                i += 1
            except MemoryError:
                print('Caught MemoryError Exception...\n')
                print(f'num = {num}, a = {a}, num = {b}, i = {i}')
                self.euler_phi(num, a, count_not_a_multiple)
        elif a is None:
            del a
        try:
            for a in range(i, num, 1):
                b = self.gcd(a, num)
                if b == 1:
                    # print(f'a = {a}, num = {num}')
                    count_not_a_multiple += 1
                    # not_a_multiple.append(a)
                    if a < num - 1:
                        # print(f'a = {a}, num = {num}')
                        wf_nop.write(f'{a}, \n')
                        # print(rf_nop.readline())
                        # print(f'{c} is not a multiple of {divsr}')
                    else:
                        # print(f'a = {a}, num = {num}')
                        wf_nop.write(f'{a}')
                        # print(rf_nop.readline())
                        # print(f'{c} is not a multiple of {divsr}')
                if b != 1:
                    count_is_a_multiple += 1
                    # is_a_multiple.append(a)
                    if a < num - 2:
                        wf_mop.write(f'{a}, \n ')
                        # print(rf_mop.readline())
                        # print(f'{a} is a multiple of {divsr}')
                    else:
                        wf_mop.write(f'{a}')
                        # print(rf_mop.readline())
                        # print(f'{a} is a multiple of {divsr}')

                i += 1
        except MemoryError:
            print('Caught MemoryError Exception...\n')
            print(f'num = {num}, a = {a}, num = {b}, i = {i}')
            # count_not_a_multiple = len(not_a_multiple)
            self.euler_phi(num, a, count_not_a_multiple)
        # is_a_multiple = ([a for a in range(divsr-1) if math.gcd(a, divsr) == 1])
        # print(f'\nInverses / Non-Multiples of [<< {num} >>] are: {not_a_multiple}\n')
        # print(f'\nMultiples / Non-Inverses of [<< {num} >>] are: {is_a_multiple}\n')

        count = 0
        val = 0
        while True:
            lines = rf_nop.read()
            val = len(lines.split(","))
            if val <= 1:
                print(val)
                break
            count += val
        print(count)
        if count_not_a_multiple > 2:
            print(f'|752| Count not a multiple: {count_not_a_multiple}'
                  f' | Count is a multiple: {count_is_a_multiple}')
            # wf_nop.write(f']; \nCount = {count_not_a_multiple}')
            # c = (num - count)
            # wf_mop.write(f']; \nCount = {count_is_a_multiple}')
        else:
            print(f'|758| Count not a multiple: {count_not_a_multiple}'
                  f' | Count is a multiple: {count_is_a_multiple}')
            # print(f'Count: {count}')
            # wf_nop.write(f']; \nCount = {count}')
            # c = (num - count)
            # wf_mop.write(f']; \nCount = {count_is_a_multiple}')
        return count_not_a_multiple

    ###############################################################################

    def powers_of_2(self, power=None) -> list:
        """This method return the values of 2 decimal conversion of the given number.

            This method converts the number to binary, converts each ON ["1"]
            point to its decimal representation, and returns an array of
            converted ON point.
                a ≡ r (mod m) | where r is your residue for each iteration
                of [ a=1, a=2, ..., m - 1 ] where gcd(a, m) == 1
        :param self:
        :param power: The number used to find residue values
        :return: array of values
        """
        if power is not None:
            pwr = power
        elif self.pwr is not None:
            pwr = self.pwr
        bin_value = bin(pwr)  # convert our number to binary
        length = bin_value.__len__()  # get the length of our binary number
        bin_value = bin_value[2:length]  # split the first two digits
        length = bin_value.__len__()  # get the length of our binary number
        bin_array1 = []
        # print(bin_array1)
        for x in bin_value:
            bin_array1.append(int(x))
            length -= 1
        # print(bin_array1)
        bin_array1.reverse()  # reverse the array before we check values
        # print(bin_array1)
        exp = 1  # 2**exp;  this is the exponent of our 2's power
        solutions = []
        lt = 0
        for c in bin_array1:
            lt += 1
            if c == 1:
                solutions.append(exp)  # assign the 2**n value of our bit
            exp += exp  # increment the exponent after each iteration
        bin_array1.reverse()  # reverse back to standard notation
        solutions.reverse()
        # print(f'\nBinary representation of [ {pwr} ] is:\n {bin_array1}\n {solutions}')
        solutions.reverse()
        # print(f'\nThe POWER [ {pwr} ], binary to decimal
        # conversion ON ["1"] bits are:\nWhere values =  {solutions}\n')
        return solutions

    ###############################################################################

    def standard_residue(self, power=None, dividend=None, divisor=None) -> array:
        """This function calculates the standard residue of an integer n, such that
            [  dividend**values ≡ remainder (modulo divisor)  ]

            We take an array of values of 2 and calculate the values of
            [ dividend**values ≡ remainder (modulo divisor) ],
            where i = number of elements in the array starting at index 0, and
            k = each integer value power of 2 in our array, and m = the standard
            residue for each iteration.
        :param self:
        :param power: The values of 2 array
        :param dividend:   The integer base of power
        :param divisor:   The modulo integer used to calculate our standard residue
        :return:    An array of the standard residue
        """
        if power is not None and dividend is not None and divisor is not None:
            pwr = power
            divid = dividend
            divsr = divisor
        elif self.pwr is not None and self.dividend is not None and self.divisor is not None:
            pwr = self.pwr
            divid = self.dividend
            divsr = self.divisor
        else:
            raise AttributeError(f'Self is null...You must provide the given parameters...')
        i = 0
        s = []
        pwrs = self.powers_of_2(pwr)
        number_of_elements = len(pwrs)
        print('\n')
        while i < number_of_elements:
            rem = pow(divid, pwrs[i], divsr)
            s.append(rem)
            # print(f'The standard residue of [ {divid} ** {pwrs[i]} ≡ {rem} (mod {divsr}) ] is: \n {s}\n')
            i += 1

        print(f'The standard residue of [ {divid}', f'{pwr}'.translate(self.super_trans),
              f' ≡ r (mod {divsr}) ] is: \n {s}\n')
        return s

    ###############################################################################

    def fast_powers(self, power=None, dividend=None, divisor=None) -> None:
        """Given two positive numbers, a and n, a modulo n (abbreviated as a mod n)
           is the remainder of the Euclidean division of a by n, where a is the
           dividend and n is the divisor.

        :param self:
        :param power:
        :param dividend:
        :param divisor:
        :return:
        """
        if power is not None and dividend is not None and divisor is not None:
            pwr = power
            divid = dividend
            divsr = divisor
        elif power is not None and self.dividend is not None and self.divisor is not None:
            pwr = power
            divid = self.dividend
            divsr = self.divisor
        elif self.pwr is not None and self.dividend is not None and self.divisor is not None:
            pwr = self.pwr
            divid = self.dividend
            divsr = self.divisor
        else:
            raise AttributeError(f'Self is null...You must provide parameters...')
        val = self.powers_of_2(pwr)
        xx = 0
        print(f'\nFast Powers Method: \n')
        while xx < val.__len__():
            tt = (divid ** val[xx]) % divsr
            print(f'{divid}', f'{val[xx]}'.translate(self.super_trans), f' ≡ {tt} (mod {divsr})')
            xx += 1

    ###############################################################################

    def fermat_little_theorem(self=None, divisor=None) -> bool:
        """A positive integer n > 1 is said to be prime if for every integer 1 < d < n, d does not divide n.
        Otherwise, n > 1 is said to be composite. For various reasons, n = 1 is neither prime nor composite
        (n = 1 is called a unit).

        - [ a**p ≡ a (mod p) ]
        - Fermat’s “Little” Theorem says that if the positive integer p is prime number, then a**p ≡ a (mod p)
        We won’t prove this just yet. A consequence of this result is that if an 6≡ a (mod n), then the
        positive integer n > 1 cannot be prime! Our previous computation uses a = 3 to show that n = 2021 is
        not prime, since 759 6≡ 3 (mod 2021).

        :param self:
        :param divisor:
        :return:
        """
        if divisor is not None:
            p = divisor
        else:
            p = self.divisor
        a = 1
        is_prime = 0
        while is_prime == 0:
            if pow(a, p, p) != a:
                is_prime = 1
                break
            a += 1
        return is_prime

    ###############################################################################

    def first_order(self, dividend=None, divisor=None):
        """The Order function finds the first e = exponent, when applied to:
            - b = base
            - n = mod
            - e = exponent
            - b**e ≡ 1 (mod n) || b**e % n = 1
            - we find all values of e that makes [ b**e ≡ 1 (mod n) ] = True

        :param self:
        :param dividend:
        :param divisor:
        :return:
        """
        if dividend is not None and divisor is not None:
            a = dividend
            n = divisor
            phi = n - 1
        else:
            a = self.dividend
            n = self.divisor
            phi = n - 1
        odr = []
        prime_factors = [1]+self.get_keys(factorint((n-1)))+[phi]
        print(prime_factors)
        for e in prime_factors:
            if pow(a, e, n) == 1:
                odr.append(e)
                # print(f'Found {e}')
        if odr.__len__() == 0:
            for i in range(phi, n**n):
                if pow(a, i, n) == 1:
                    odr.append(i)
                    # print(f'Found {i}')
                    break
        return odr

    ###############################################################################

    def get_orders(self=None, values=list, v=None) -> list:
        """The function returns the divisors grouped and sorted for all values included
            in the input set.

        :param self:
        :param v: add [ v | v+ ] marker for verbose output
        :param values: a single value or list of values
        :return:
        """
        if values is list:
            raise SyntaxError(f"""Please use list format: get_orders([52]) for single value or 
            get_orders([52, 35, 95]) for multiple values...""")
        elif values is not None and self is not None:
            list_of_num = values
            # print(f'{num_list}')
            print(f'############################################################################################')
        else:
            print('Exiting......!!!\nCheck your input......!!!\n')
            exit(0)
        sol = []
        i = 1
        for num in list_of_num:
            div_list = self.get_divisors(num)  # Get the divisors of each number
            if v is not None:
                if v == 'v' or v == "v+":
                    print(f'Divisors for {num} are: {div_list}')  # Print the list of divisors
                elif v != 'v' or v != "v+" and i == 1:
                    raise SyntaxWarning("""You entered the incorrect option......!!!
                For verbose output, enter < "v" or "v+" >""")
            sol += div_list  # append the list shorthand
            i += 1
            # diff += set(sol).symmetric_difference(set(div_list))    # append the list using set()
        sol.sort()  # sort the list
        sol1 = [a for a in sol if (sol.count(a) > 1)]
        print(f'Shared Solution Set: {sol1}')
        sol2 = [a for a in sol if (sol.count(a) == 1)]
        print(f'Negation Solution Set: {sol2}')
        # for a in sol:
        #     if sol.count(a) == 1:
        #         sol.remove(a)  # remove duplicates based on number of occurrences
        # print(f'Solution: {sol}')
        print(f'############################################################################################')
        return sol

    ###############################################################################

    def euler_totient(self, number=int):
        """This function calculates φ(n) = phi; it finds the value of φ(n) = phi using euler's
            totient formula.

            Basic idea:
                - n = the number in phi of n [ φ(n) ]
                - f = factors of n
                - k = list of integers from [ 1 ≤ k ≤ n ]

                - if φ(n) = phi = n−1:
                then n is prime; & all values of k (1<k<n-1) is  a**k ≡ 1 (mod n),
                [ a**k is congruent to 1 (mod n) ]; and if n is prime no values will be
                eliminated from our set of divisors. In other words, a positive integer n
                is prime if and only if φ(n) = n−1.

                - if n is not prime:
                1. Start with a list of integers from [ 1 ≤ k ≤ n ]
                2. remove or cross out all prime factors of n in our list of [ 1 ≤ k ≤ n ]

                3. count the total numbers remaining elements in your list of [ 1 ≤ k ≤ n ]
                    - φ(n) = phi = [ [phi Σ {k1, ..., kn} ] = phi ]

        :param number:
        :return:
        """
        if number is not int and self is not None:
            n = number
            f = factorint(n)  # prime factors
        else:
            print(f'Exiting..........!!!\nCheck you values and try again..........!!!\n')
            exit(0)
        pool = [a for a in range(1, n + 1)]  # create a pool of number from 1 to n
        for a in f:  # for each prime factor
            while True:  # remove all multiples of that number from our pool
                pool = [j for j in pool if (j % a) != 0]
                break  # stop when we finished removing all multiples of our prime factors
        # print(f'pool: {pool}')
        c = count(pool)  # lastly we count the numbers that is left and return the count
        # print(f'count: {c}')
        return c

    ###############################################################################

    def get_keys(self, dictionary=dict) -> list:
        """Gets a list of keys in the dictionary.

        :param dictionary:
        :return: list of keys
        """
        if self is not None:
            keys = []
            for key in dictionary.keys():
                keys.append(key)
            # print(keys)
        return keys

    ###############################################################################

    def order_of_element(self, dividend=None, divisor=None, negation=None) -> list:
        """Order of an element (mod n): For fixed integers a and n, with gcd(a,n) = 1,
        consider the congruence a**k ≡ 1 (mod n).  If an integer k = h satisfies this,
        then so does k = hm:

            - Fermat's Little Theorem (v2) says: if n is prime and a is not a multiple of n
                [ a !≡ 0 (mod n) ], then a**n-1 ≡ 1 (mod n)
            - a**hm ≡ (a**h)**m ≡ 1**m ≡ 1 (mod n)
            - ord(a) is the smallest exponent(s) [ a**h, a**hm, (a**h)**m, 1**m ], that
              makes the statement [ a**h, a**hm, (a**h)**m, 1**m ] congruent to 1 (mod n)
            - the exponent we refer to as: [ h, hm, h**m, m ]
                - [ a**h, a**hm, (a**h)**m, 1**m ] ≡ 1 (mod n)

        If gcd(a,n) = 1, the order of a modulo n is the smallest positive integer h such
        that a**h ≡ 1 (mod n).

        Our notation:
            - ord(a) = h, if a**h ≡ 1 (mod n)
            - If gcd(a,n) > 1, ord(a) is not defined

        For any positive integer k, the following are logically equivalent:
            - a**k ≡ 1 (mod n)
            - k is a multiple of ord(a)
            - ord(a) is a factor of k
            - This means that if a**k !≡ 1 (mod n), then ord(a) IS NOT a factor of k.

        :param dividend:
        :param divisor:
        :param negation: use [ n |-n ] flag to return negation or ( !≡ ) values
        :return:
        """
        if dividend is not None and divisor is not None and self is not None:
            a = dividend
            n = divisor
        else:
            print(f'Exiting..........!!!\nCheck you values and try again..........!!!\n')
            exit(0)
        phi = self.euler_totient(n)  # get our phi of n using euler's totient
        # print(f'PHI: {phi}')
        sol = []  # capture our solution set in a list
        for k in range(1, n + 1):
            if negation == 'n' or negation == '-n':
                if self.gcd_euclid(a, n) == 1 and pow(a, k, n) != 1 and k != 1:    # find values congruent to a**k ≡ 1 (mod n)
                    if (k >= phi) and (k % phi) == 0:  # k is a multiple of ord(a)
                        sol.append(k)
                        print(f'Found [h > phi]: {k}')
                    elif (phi >= k) and (phi % k) == 0:  # k is a multiple of ord(a)
                        sol.append(k)
                        print(f'Found [phi > h]: {k}')

            elif negation is None:
                if self.gcd_euclid(a, n) == 1 and pow(a, k, n) == 1 and k != 1:  # find values not-congruent to a**k !≡ 1 (mod n)
                    if (k >= phi) and (k % phi) == 0:  # k is a multiple of ord(a)
                        sol.append(k)
                        # print(f'Found [h > phi]: {k}')
                    elif (phi >= k) and (phi % k) == 0:  # k is a multiple of ord(a)
                        sol.append(k)
                        # print(f'Found [phi > h]: {k}')
            elif negation != 'n' or negation != '-n' and negation is not None:
                raise SyntaxWarning(f'use [ n | -n ] flags for negation or non-congruent values......!!!')
        # print(f'Order of elements Solutions: \n{sol}')
        return sol

    ###############################################################################

    def get_divisors(self=None, number=None):
        """This method accepts a number and returns list of integers that divides
            our number.

        - for each integer a in range(start integer, stop integer) that has a remainder of 0 when, [ number % a = 0 ]

        :param self:
        :param number:
        :return:
        """
        if number is not None:
            b = number
        else:
            b = self.pwr
        q = lambda a, num: num % a
        lot = [a for a in range(1, b + 1) if q(a, b) == 0]
        return lot

    ###############################################################################

    def order_v2(self, ord=None, divisor=None) -> any:
        """

        :param ord:
        :param divisor:
        :return:
        """
        if ord is not None and divisor is not None and self is not None:
            # define the phi = n - 1
            n = divisor
            phi = n - 1
            # get a list of all the prime factors of your phi
            # expand the list to find all factors of phi
            prime_factors = self.get_keys(factorint(phi))
            p_f = prime_factors.copy()
            p_f.reverse()
            list_of_divisors = [phi]+[(phi//k) for k in prime_factors]+[k for k in p_f]+[1]
            print(f'Factors of {phi}: {list_of_divisors}')
            # use each value in your list of factors to verify that a smaller value
            # exists than [ n - 1 ], that is congruent to 1 (mod n).  Proving that
            # n is not prime
            order = [k for k in list_of_divisors if pow(ord, k, n) == 1]
            # But if all values fail to prove congruence that without a show of a
            # we can say now that n is surely prime
            order.reverse()
            if order.__len__() == 1 and order[0] == n - 1:
                # print(f'Order({ord}) = {order}')
                # if only one value is returned and that value is equivalent to
                # (n - 1) then we can say that n is prime for sure
                # print(f'|[...YAW...]| {n} is PRIME!!!')
                return True, order, f'{n} is prime'
            elif order.__len__() > 1:
                return False, order, f'{n} is not prime'
            elif order.__len__() == 0:
                inf = n**n
                for k in range(1, inf):
                    if pow(ord, k, n) == 1:
                        order.append(k)
                        break
                # print(f'Order({ord}) = {order}')
                # print(f'|[...NAW...]| {n} is NOT-PRIME!!!')
                return False, order, f'{n} is not prime'

    ###############################################################################

    def pratt(self, dividend=None, divisor=None) -> str:
        """This information is called a Pratt Certificate of primality.

        - a = dividend
        - n = divisor
        - phi = n - 1
        - f = "prime" factors of (phi)

        To show that [ n = divisor ] is prime, we consider the prime factorization of φ(n) = phi = n−1:
        It can be shown that any proper divisor of [ φ(n) = phi = n−1 ] must divide [ (n−1)//(pi) ] for
        at least one 1 ≤ i ≤ size of "f". So we need to find a value of "a" for which:

        - a ** n−1 ≡ 1 (mod n) [n passes Fermat’s Test]
        - a ** ((n−1)/{pi, ..., pn}) !≡ 1 (mod n) [ must be TRUE] for EVERY prime factor of φ(n) =  phi = n-1

         This suggests that ord(a) = n−1 or phi, and thus "n" is prime. Knowing the prime
         factorization of phi or n−1, along with such a value of a, allows anyone to verify
         that "n" is prime.

        :param dividend: dividend
        :param divisor: divisor
        :return:
        """
        if self is None or self is not None and dividend is not None and divisor is not None:
            a = dividend
            n = divisor
            phi = n - 1
        prime_factors = self.get_keys(factorint(phi))
        print(f'The prime factorization of {phi} is {factorint(phi)}')
        sol = []
        # verify that: a ** n-1 ≡ 1 (mod n) ==> True
        for k in prime_factors:
            # verify that: [ a ** ((n-1)/pi, ..., pn) !≡ 1 (mod n) ] is not
            # congruent to !≡ 1 (mod n) for every prime divisor [ pi, ..., pn ] of
            # [ n - 1 ].
            if pow(a, (phi//k), n) != 1 and pow(a, phi, n) == 1:
                sol.append(k)
                print(f'Found: {a} ** ({phi}//{k}) ≡ {pow(a,(phi//k), n)} !≡ 1 (mod {n})')
        if sol.__len__() > 1:
            return f'{n} is not PRIME...!!!'
        else:
            return f'{n} is PRIME...!!!'

    ###############################################################################

    def restricted_cancel_law(self, cc=None, dividend=None, divisor=None):
        """The Restricted Cancellation Law:

        Let [ ad ≡ bd (mod c) ] and [ gcd(c, d) = 1 ]; by definition:

            - ad ≡ bd (mod c)   # we are allowed to cancel a number from both sides of a
                                # congruence, as long as it has gcd 1 with the modulus
            - c | (ad - bd)     # we can factor out and cancel "d" on both sides
            - c | d(a - b)      # we now have a simplified version
            - c | (a - b)       #
            - ax + by = c       # ** has solutions with x and y integers if and only if c is
                                # a multiple of gcd(a,b) **
                                # we can find solutions using Knuth's GCD algorithm
            - d = gcd(a, b)     # their exist integers (x, y) such that ax + by = d
            - (a/d)x + (b/d)y = 1  # 1 is multiple of gcd((a/d), (b/d)) = 1

            Once you’ve found one solution (x0, y0), to [ ax + by = c ], there is a simple formula
            that gives every solution (x, y). Starting with:

            - ax + by = c
            - ax0 + by0 = c
            - a(x - x0) + b(y - y0) = 0   # subtraction
            - (a/d)(x - x0) = - (b/d)(y - y0)  # rearrange and divide by d = gcd(a, b); and d > 0

            If gcd((a/d),(b/d)) = 1, then [ (b/d) | (x - x0) ], thus (x - x0) = t(b/d) and (y - y0)
            = t(a/d) for some integer "t", giving a solution of:
            - x = x0 + t(b/d) = x0 + t(b)
            - y = y0 - t(a/d) = y0 - t(a)

            Ex:  17x + 27y = 5048       # First use Knuth to find
            - v1 = array([1, 0, 27])    # notice we placed the larger of the two values first
            - v2 = array([0, 1, 17])    # because 17//27 = 0; the quotient will be a decimal number < 0
            - 27//17 = 1                # we only want the integer portion of our results, toss the decimal portion
            - v3 = v1 - 1*v2 = array([-1,  1, 10])
            - 17/10 = 1
            - v4 = v2 - 1*v3 = array([ 2, -1,  7])
            - 10/7 = 1
            - v5 = v3 - 1*v4 = array([-3,  2,  3])
            - 7/3 = 2
            - v6 = v4 - 2*v5 = array([ 8, -5,  1])
            - 17*(8) + 27*(-5) = 1      # we found our (x, y), using the gcd(17, 27)
            - x = (8*5048) = [ 40384 ], y = (-5*5048) = [ -25240 ]  # correct values of (x, y)
            - 17*(8*5048) + 27*(-5*5048) = 5048     # we prove our [ x, y ] values
            - A particular (x, y) solution for t = 2: ??? IS "t" A RANDOM PRIME OR RANDOM NUMBER ???
                - x = x0 + t(b/d) = 40384 + 2*(27) = 40438
                - y = y0 - t(a/d) = -25240 - 2*(17) = -25274
                - For x to be non-negative, "t" must be greater than or equal to: [LOWER BOUNDS]
                    - x = x0
                    - (-x0 // (b/d)) = (-40384//27) = -1495.7037037037037 = -1495
                - For y to be non-negative, "t" must be less than or equal to: [UPPER BOUNDS]
                    - (-y0 // (a/d)) = (-25240//17) = -1484.7058823529412 = -1484
                - These values of "t" give a solution with non-negative x and y:
                    - list(range(-1495,-1484)):
                    [-1495, -1494, -1493, -1492, -1491, -1490, -1489, -1488, -1487, -1486, -1485]
                - Now we can find EVERY pair of positive integers (x, y), such that [ ax + by = d ]
                    - >>> for t in range(-1495, -1484):
                      ...     print(f'y = 40384 + 27*{t} = {40384+27*t},  y = -25240 - 17*{t} = {-25240-17*t}')
                    - y = 40384 + 27*-1495 = 19,  y = -25240 - 17*-1495 = 175
                    - y = 40384 + 27*-1494 = 46,  y = -25240 - 17*-1494 = 158
                    - y = 40384 + 27*-1493 = 73,  y = -25240 - 17*-1493 = 141
                    - y = 40384 + 27*-1492 = 100,  y = -25240 - 17*-1492 = 124
                    - y = 40384 + 27*-1491 = 127,  y = -25240 - 17*-1491 = 107
                    - y = 40384 + 27*-1490 = 154,  y = -25240 - 17*-1490 = 90
                    - y = 40384 + 27*-1489 = 181,  y = -25240 - 17*-1489 = 73
                    - y = 40384 + 27*-1488 = 208,  y = -25240 - 17*-1488 = 56
                    - y = 40384 + 27*-1487 = 235,  y = -25240 - 17*-1487 = 39
                    - y = 40384 + 27*-1486 = 262,  y = -25240 - 17*-1486 = 22
                    - y = 40384 + 27*-1485 = 289,  y = -25240 - 17*-1485 = 5
        :return:
        """
        if cc is not None and dividend is not None and divisor is not None and self is not None:
            a = divisor
            b = dividend
            c = cc
        else:
            a = self.divisor
            b = self.dividend
            c = self.pwr

    ###############################################################################
    # ############################ END OF CLASS GCD ############################# #
    ###############################################################################


if __name__ == "__main__":
    ###############################################################################
    #   Enter values below you want to test:
    ###############################################################################
    x_n_divisor = 8321
    y_m_dividend = 29
    e_pwr = 0
    z_quotient = 31
    remainder = 0
    ###############################################################################
    ###############################################################################
    gcd = GCD(y_m_dividend, x_n_divisor, e_pwr)
    # print(gcd.euclid())
    # print(gcd.euclid(y_m_dividend, x_n_divisor))
    # print(gcd.knuth())
    # gcd.knuth_verbose()
    # print(gcd.knuth())
    # print(gcd.pwrss())
    # print(gcd.values())
    # gcd.values()
    # gcd.knuth_streamlined()
    # gcd.knuth_solution(1)
    # gcd.knuth_solution(z_quotient, y_m_dividend, x_n_divisor)
    # print(gcd.powers_of_2())
    # print(gcd.standard_residue())
    # gcd.standard_residue(1398168, 491, 1398169)
    # gcd.fast_powers(2041, 6884, 95489)
    # print(gcd.euler_phi(788533))
    # print(gcd.fermat_little_theorem(7663683516910153))
    # print(phi_interval(0, 5))
    # print(gcd.get_divisors(16))
    # gcd.get_orders([61588, 178, 173], "v")
    # print(gcd.euler_totient(407))
    # print(gcd.first_order(729, 852353))
    # print(gcd.order_of_element(2, 11))  # enter o_o_e(a,b,"n") for negation
    # print(gcd.order_of_element(729, 852353))
    # print(gcd.order_v2(729, 852353))
    # print(gcd.pratt(729, 852353))
    # print(gcd.restricted_cancel_law())
    # solt = [a for a in gcd.get_orders([5088, 7008]) if gcd.gcd_euclid(a, 7008) == 1]
    # print(solt)
    ###############################################################################
    ###############################################################################
    #     The below statement uses math.gcd(a,num) to validate euclid() method.   #
    ###############################################################################

    # print(f"""\nPython GCD built-in function calculated the GCD of
    # "{t_dividend}" & "{t_divisor}" is: {math.gcd(t_dividend, t_divisor)}.\n\n""")

    ###############################################################################
    ###############################################################################

    # Rule: if gcd(a,n)==1 => smallest positive integer h
    # a**h ≡ 1 (mod n)  | pow(a,h,n)==1
    # n is the order of a (mod n)
    # find 'a' with order = 866
    # a**k ≡ 1 (mod n)  <==> k is a multiple of order(a)
    # 2**886 ≡ 1 (mod 887)
    # 886 is a multiple of order(a)
    # thus order(a) is a factor/divisor of 886
    # 2**1 (!≡) 1 (mod 887)  <==> (1) is not a multiple of order(a)

    # This says order(2) is not a factor of the exponent 1
    # print(pow(2, 1, 887))
    # # expected result: 2
    # # This says order(2) is not a factor of the exponent 2
    # print(pow(2, 2, 887))
    # # expected result: 4
    # aaa = 2
    # n = 887
    # # the order of a (mod n)
    # # To prove that 887 is prime:
    # print(2*443)
    # # == 886
    # print(pow(97, 886, 887))   # = 1
    # print(pow(97, 2, 887))     # = 539
    # print(pow(97, 443, 887))   # = 886
    # # To prove 443 is prime...
    # print(factorint(442))
    # print(2*13*17)
    # print(pow(44, 442, 443))
    # print(divisors(26))
    # p = 139187
    # print(pow(2, p-1, p))
    # print(pow(3, p-1, p))
    # print(pow(44, p-1, p))
    # print(pow(2, (p-1)//2, p))
    # print(pow(2, (p - 1) // 69593, p))
    # print(divisors(p-1))
    # if [ a**p ≡ 1 (mod n) ] we have a power that is congruent to 1 the order a (mod n) is a factor of our power
    # but if [ a**p !≡ 1 (mod n)] we have a power that is not congruent to 1, order a is not a factor of our power

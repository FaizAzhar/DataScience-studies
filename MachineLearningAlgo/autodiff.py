from abc import ABC, abstractmethod
import math


class DifferentiableSymbolicOperation(ABC):
    
    # Differentiate with respect to 'var'
    @abstractmethod
    def backward(self, var):
        pass
    
    # To return the gradient at specific point
    @abstractmethod
    def compute(self):
        pass
    
    # To return the equation of gradient
    @abstractmethod
    def simplify(self):
        pass


def with_symbolic_constant(fn):
    def wrapper(self, other):
        if not isinstance(other, DifferentiableSymbolicOperation):
            other = Const(other)
        return fn(self, other)
    return wrapper
    

class DifferentiableSymbolicOperatorsMixin:
    @with_symbolic_constant
    def __add__(self, other):
        return Sum(self, other)
    
    @with_symbolic_constant
    def __radd__(self, other):
        return Sum(other, self)
    
    @with_symbolic_constant
    def __sub__(self, other):
        return Sub(self, other)
    
    @with_symbolic_constant
    def __rsub__(self, other):
        return Sub(other, self)
    
    @with_symbolic_constant
    def __mul__(self, other):
        return Mul(self, other)
    
    @with_symbolic_constant
    def __pow__(self, other):
        return Pow(self, other)
    
    @with_symbolic_constant
    def __rpow__(self, other):
        return Pow(other, self)
    
    @with_symbolic_constant
    def __rmul__(self, other):
        return Mul(other, self)
    
    @with_symbolic_constant
    def __truediv__(self, other):
        return Div(self, other)
    
    @with_symbolic_constant
    def __rtruediv__(self, other):
        return Div(other, self)
    
    def __neg__(self):
        return Mul(Const(-1), self)



class Const(DifferentiableSymbolicOperation, DifferentiableSymbolicOperatorsMixin):
    def __init__(self, value):
        self.value = value
    
    def backward(self, var):
        return 0

    def compute(self):
        return self.value
    
    def simplify(self):
        return self
    
    def __eq__(self, other):
        if isinstance(other, Const):
            return self.value == other.value
        else:
            return self.value == other
        
    def __repr__(self):
        return str(self.value)


class Var(DifferentiableSymbolicOperation, DifferentiableSymbolicOperatorsMixin):
    def __init__(self, name, value=None):
        self.name, self.value = name, value
    
    def backward(self, var):
        return 1 if self == var else 0

    def compute(self):
        if self.value is None:
            raise ValueError('unassigned variable')
        return self.value
    
    def simplify(self):
        return self
    
    def __repr__(self):
        return f'{self.name}'
        if self.value is not None:
            return f'{self.name}({self.value})'
        else:
            return f'{self.name}'

            
class Sum(DifferentiableSymbolicOperation, DifferentiableSymbolicOperatorsMixin):
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def backward(self, var):
        return self.x.backward(var) + self.y.backward(var)

    def compute(self):
        return self.x.compute() + self.y.compute()
    
    def simplify(self):
        x, y = self.x.simplify(), self.y.simplify()
        if x == 0:
            return y
        elif y == 0:
            return x
        elif isinstance(x, Const) and isinstance(y, Const):
            return Const(x.value + y.value)
        else:
            return x + y
    
    def __repr__(self):
        return f'({self.x} + {self.y})'


class Sub(DifferentiableSymbolicOperation, DifferentiableSymbolicOperatorsMixin):
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def backward(self, var):
        return self.x.backward(var) - self.y.backward(var)

    def compute(self):
        return self.x.compute() - self.y.compute()
    
    def simplify(self):
        x, y = self.x.simplify(), self.y.simplify()
        if x == 0:
            return y
        elif y == 0:
            return x
        elif isinstance(x, Const) and isinstance(y, Const):
            return Const(x.value - y.value)
        else:
            return x - y

    def __repr__(self):
        return f'({self.x} - {self.y})'


class Mul(DifferentiableSymbolicOperation, DifferentiableSymbolicOperatorsMixin):
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def backward(self, var):
        return self.x.backward(var) * self.y + self.x * self.y.backward(var)

    def compute(self):
        return self.x.compute() * self.y.compute()
    
    def simplify(self):
        x, y = self.x.simplify(), self.y.simplify()
        if x == 0 or y == 0:
            return 0
        elif x == 1:
            return y
        elif y == 1:
            return x
        elif isinstance(x, Const) and isinstance(y, Const):
            return Const(x.value * y.value)
        else:
            return x * y

    def __repr__(self):
        return f'({self.x} * {self.y})'

class Pow(DifferentiableSymbolicOperation, DifferentiableSymbolicOperatorsMixin):
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def backward(self, var):
        if isinstance(self.y, Const):
            return self.y * (Pow(self.x,self.y-1)) * self.x.backward(var)
        return self.y * (Pow(self.x,self.y-1)) * self.x.backward(var) + Pow(self.x,self.y) * self.y.backward(var)* Log(self.x) 

    def compute(self):
        return math.pow(self.x.compute(),self.y.compute())
    
    def simplify(self):
        x, y = self.x.simplify(), self.y.simplify()
        if y == 0:
            return 1
        elif isinstance(y, Const):
            return x**y.value
        elif isinstance(x, Const):
            return Const(math.log(x.value)) * (x**y)
        else:
            return Pow(x,y)

    def __repr__(self):
        return f'({self.x} ** {self.y})'

class Div(DifferentiableSymbolicOperation, DifferentiableSymbolicOperatorsMixin):
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def backward(self, var):
        return (
            self.x.backward(var) * self.y - self.x * self.y.backward(var)
        ) / (self.y * self.y)

    def compute(self):
        return self.x.compute() / self.y.compute()
    
    def simplify(self):
        x, y = self.x.simplify(), self.y.simplify()
        if x == 0:
            return 0
        elif y == 1:
            return x
        elif isinstance(x, Const) and isinstance(y, Const):
            return Const(x.value / y.value)
        else:
            return x / y

    def __repr__(self):
        return f'({self.x} / {self.y})'


class Exp(DifferentiableSymbolicOperation, DifferentiableSymbolicOperatorsMixin):
    def __init__(self, x):
        self.x = x
    
    def backward(self, var):
        return Exp(self.x) * self.x.backward(var)

    def compute(self):
        return math.exp(self.x.compute())
    
    def simplify(self):
        x = self.x.simplify()
        if x == 0:
            return 1
        elif isinstance(x, Const):
            return Const(math.exp(x.value))
        else:
            return Exp(x)

    def __repr__(self):
        return f'exp({self.x})'


class Log(DifferentiableSymbolicOperation, DifferentiableSymbolicOperatorsMixin):
    def __init__(self, x):
        self.x = x
    
    def backward(self, var):
        return self.x.backward(var) / self.x

    def compute(self):
        return math.log(self.x.compute())
    
    def simplify(self):
        x = self.x.simplify()
        if x == 1:
            return 0
        elif isinstance(x, Const):
            return Const(math.log(x.value))
        else:
            return Log(x)

    def __repr__(self):
        return f'log({self.x})'

class Sin(DifferentiableSymbolicOperation, DifferentiableSymbolicOperatorsMixin):
    def __init__(self, x):
        self.x = x
    
    def backward(self, var):
        return Cos(self.x) * self.x.backward(var)

    def compute(self):
        return math.sin(self.x.compute())
    
    def simplify(self):
        x = self.x.simplify()
        if isinstance(x, Const):
            return Const(math.sin(x.value))
        else:
            return Sin(x)

    def __repr__(self):
        return f'sin({self.x})'

class Cos(DifferentiableSymbolicOperation, DifferentiableSymbolicOperatorsMixin):
    def __init__(self, x):
        self.x = x
    
    def backward(self, var):
        return -Sin(self.x) * self.x.backward(var)

    def compute(self):
        return math.cos(self.x.compute())
    
    def simplify(self):
        x = self.x.simplify()
        if isinstance(x, Const):
            return Const(math.cos(x.value))
        else:
            return Cos(x)

    def __repr__(self):
        return f'cos({self.x})'

class Tan(DifferentiableSymbolicOperation, DifferentiableSymbolicOperatorsMixin):
    def __init__(self, x):
        self.x = x
    
    def backward(self, var):
        return 1/(Cos(self.x)**2) * self.x.backward(var)

    def compute(self):
        return math.tan(self.x.compute())
    
    def simplify(self):
        x = self.x.simplify()
        if isinstance(x, Const):
            return Const(math.tan(x.value))
        else:
            return Tan(x)

    def __repr__(self):
        return f'tan({self.x})'
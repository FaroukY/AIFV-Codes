import numpy as np
from math import log2
import collections 
from matplotlib import pyplot as plt

"""
Basic BstNode for pretty printing
"""
class BstNode(object):
    def __init__(self, key):
        self.key = key
        self.right = None
        self.left = None
        self.leaf = False
        self.master = False
        self.complete = False
        self.slave = False
        self.specialInT1 = False

    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = '%s' % self.key
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.key
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.key
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.key
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2


Wm = None

"""
This precalculates Wm[i] = \sum_{j=0}^i {p[i]}
"""
def precalc_Wm(p):
    global Wm 
    Wm = [0 for _ in range(len(p)+1)]
    for i in range(1, len(p)+1):
        Wm[i] = Wm[i-1]+p[i-1]

#Uses WM to  return Wm[m]
def W(m):
    global Wm
    if m<0:
        return 0
    if m>= len(p):
        return 1
    return Wm[m]

#1 - W(m)
def WPrime(m):
    global Wm
    if m<0:
        return 1
    return 1-W(m)

# W(mprime) - W(m)
def WTwo(m, mprime):
    if m>=mprime:
        return 0
    return W(mprime)-W(m)

"""
This is a helper function. It looks at how many leaves, master nodes are left on the current level
Then assigns the child node either a leaf or master (or makes it internal node) depending on the number left
Used in both constructT0 and constructT1
"""
def assign_child(leaves, masters, Pis, idx_leaf, idx_mast, cur_depth):
    child = None
    lt = 0
    new_q1t0 = 0
    new_q0t1 = 0
    if leaves>0:
        child = BstNode(Pis[idx_leaf])
        lt += Pis[idx_leaf]*(cur_depth + 1)
        new_q0t1 += Pis[idx_leaf]
        child.depth = cur_depth + 1
        child.leaf = True
        leaves = leaves - 1
        idx_leaf = idx_leaf + 1
    elif masters > 0:
        child = BstNode(Pis[idx_mast])
        lt += Pis[idx_mast]*(cur_depth + 1)
        new_q1t0 += Pis[idx_mast]
        child.depth = cur_depth + 1
        child.master = True
        masters = masters - 1
        idx_mast = idx_mast + 1
    else:
        child = BstNode(0)
        child.depth = cur_depth + 1
        child.complete = True
    
    return child, leaves, masters, idx_leaf, idx_mast, lt, new_q1t0, new_q0t1



"""
This is the Dynamic programming algorithm for Constructing T1 from https://arxiv.org/pdf/2001.11170.pdf
Inputs:
    p: Probability distribution. List or numpy array
    C: The constant C from the paper

Returns:
    LT1: The expected codeword length of T1^*
    q0T1: The sum of leaf probabilities in T1^*
    root: A BstNode root of the tree T_1^* that offers .display() API to print it
"""
def constructT1(p, C):
    
    P = p
    precalc_Wm(p) #Calculate Wm = \sum_{i=1}^m p[i]. 
    
    n = len(p)
    I1 = [(0,3,0),(1,1,0), (1,1,1)] #initial starting signatures. See figure 6 in https://arxiv.org/pdf/2002.09885.pdf
    
    Adj = {} #Adjacency matrix in Signature graph
    
    """
    The following code will BFS starting from signatures in I1
    For each current signature s in the BFS tree, we will consider all the signatures s' that it can expand to
    Then for each (s, s') pair, we will calculate the cost of expanding from s to s'
    Then We set Adj[s][s'] = cost to expand from s to s'
    """
    q = collections.deque(I1)
    while len(q)>0:
        s = q.popleft()
        if s not in Adj:
            Adj[s] = {}
            mp, pp, zp = s
            for e0 in range(pp+1):
                for e1 in range(pp+1):
                    if e0+e1<=pp:
                        m,p,z = mp+e0+e1, zp+2*(pp-e0-e1), e1
                        if (z>=0 and z<=m and m<=n and 0<=p and p<=n and 
                            (mp != m or pp!=p or zp!=z)):
                                next_signature = (m,p,z)
                                cost_expansion = WPrime(mp) - C*WTwo(mp, m-z)
                                Adj[s][next_signature] = cost_expansion
                                q.append(next_signature)
    
    
    all_sigs = Adj.keys()
    INF = 10*n #Infinity. AIFV-2 code length at most 4*n, put 10*n for buffer, or just any big num
    
    ShortestDist = {sig : INF for sig in all_sigs} #Shortest dist from I1 to sig
    Pred = {sig : None for sig in all_sigs} #Preceding signature to sig in path
    PredExpand = {sig : None for sig in all_sigs} #Expansion from prev sig to reach here
    
    #DP Base Cases
    ShortestDist[(1,1,0)] = 1-C*P[0] #For explanation, see Figure 6 of https://arxiv.org/pdf/2002.09885.pdf
    ShortestDist[(0,3,0)] = 1 #For explanation, see Figure 6 of https://arxiv.org/pdf/2002.09885.pdf
    ShortestDist[(1,1,1)] = 1 #For explanation, see Figure 6 of https://arxiv.org/pdf/2002.09885.pdf
    
    #Do a topological sort. I am being lazy here and just using O(nlogn) sort func, but obviously you can top sort it
    #in O(n) time using BFS/DFS. 
    topological_sort = sorted(list(all_sigs))
    
    """
    For each signature in the topological sort s
        For each signature s' that s can expand to:
            We update ShortestDist[s'] = min(ShortestDist[s] + cost to expand from s to s')
            Also save some info like e0, e1 and s that we will need later
    """
    for s in topological_sort:
        mp, pp, zp = s
        for e0 in range(pp+1):
            for e1 in range(pp+1):
                if e0+e1 <= pp:
                    m,p,z = mp+e0+e1, zp+2*(pp-e0-e1), e1
                    if (z>=0 and z<=m and m<=n and 0<=p and p<=n and 
                            (mp != m or pp!=p or zp!=z)):
                        next_sig = (m,p,z)
                        cost_expansion = WPrime(mp)-C*WTwo(mp, m-z)
                        total_cost = ShortestDist[s]+cost_expansion
                        if total_cost < ShortestDist[next_sig]:
                            ShortestDist[next_sig] = total_cost
                            Pred[next_sig] = s
                            PredExpand[next_sig] = (e0,e1)
    
    """
    Now, we can reconstruct the signatures to form the tree by tracing back from signature (n,0,0)
    Start at s, and keep going backwards using Pred dictionary
    """
    s = (n,0,0)
    signatures = [s]
    expansions = [PredExpand[s]]
    while Pred[s]:
        s = Pred[s]
        signatures = [s]+signatures
        if PredExpand[s]:
            expansions = [PredExpand[s]] + expansions
    
    
    """
    For an explanation of the following, see Figure 6 of https://arxiv.org/pdf/2002.09885.pdf
    Depending on the first signature in the path with shortest distance to (n,0,0), We construct the tree differently
    """
    root = BstNode(0)
    root.depth = 0
    root.complete = True
    root.left = BstNode("xx")
    root.left.depth = 1
    root.left.slave = True
    root.left.specialInT1 = True
    
    LT1, q0T1 = 0,0
    if signatures[0] == (1,1,1):
        root.right = BstNode(P[0])
        LT1 += P[0]
        P = P[1:]
        root.right.depth = 1
        root.right.master = True
    elif signatures[0]==(0,3,0):
        root.right = BstNode(0)
        root.right.depth = 1
        root.right.complete = 1
    elif signatures[0]==(1,1,0):
        root.right = BstNode(P[0])
        LT1 += P[0]
        q0T1 += P[0]
        root.right.depth = 1
        root.right.leaf = 1
        P = P[1:]
    else:
        raise Exception("Something went terribly wrong with the starting signature")
    
    cur_depth = 1
    current_level = [root.left, root.right]
    
    """
    The following code is a terrible Finite State machine. Please rewrite if you have the chance. 
    It starts from the initial signature, then looks at e0,e1 and constructs the Tree from top to bottom
    """
    for a,b in expansions:
        leaves = a
        masters = b
        next_level = []
        Pis = P[: leaves+masters] #These are the probabilities that will get assigned at this level
        
        idx_leaf = 0  #Probabilities Pis[0,1,...,leaves-1] will be assigned to leaves, and Pis[leaves, ..., leaves+masters-1] will be assigned to masters
        idx_mast = leaves #Probabilities Pis[0,1,...,leaves-1] will be assigned to leaves, and Pis[leaves, ..., leaves+masters-1] will be assigned to masters
        P = P[leaves+masters:]
        
        for r in current_level:
            #First depending on what kind of node r is (leaf, slave, master, or complete internal node), parse it
            #Then, depending on how many leaves/masters are left in this level, assign them appropriately. 
            if r.leaf:
                continue #Nothing else to do
            if r.slave:
                child = None           
                if leaves > 0:
                    child = BstNode(Pis[idx_leaf])
                    LT1 += Pis[idx_leaf] * (cur_depth+1)
                    q0T1 += Pis[idx_leaf]
                    child.depth = cur_depth + 1
                    child.leaf = True
                    leaves = leaves - 1
                    idx_leaf = idx_leaf + 1
                elif masters > 0:
                    child = BstNode(Pis[idx_mast])
                    LT1 += Pis[idx_mast]*(cur_depth + 1)
                    child.master = True
                    masters = masters - 1
                    idx_mast = idx_mast + 1
                else:
                    child = BstNode(0)
                    child.depth = cur_depth + 1
                    child.complete = True
                
                if r.specialInT1: #This is the 0 child of T1, so it is a slave, and must have a RIGHT child
                    r.right = child
                    next_level.append(r.right)
                else: #normal slave, set child to left. 
                    r.left = child
                    next_level.append(r.left)
                                
            if r.master:
                r.left = BstNode("xx")
                r.left.depth = cur_depth + 1
                r.left.slave = True
                next_level.append(r.left)
            
            if r.complete:
                r.left, leaves, masters, idx_leaf, idx_mast, lt1, new_q1t0, new_q0t1 = assign_child(leaves, masters, Pis, idx_leaf, idx_mast, cur_depth)
                LT1 += lt1
                q0T1 += new_q0t1
                
                r.right,leaves, masters, idx_leaf, idx_mast, lt1, new_q1t0, new_q0t1 = assign_child(leaves, masters, Pis, idx_leaf, idx_mast, cur_depth)  
                LT1 += lt1
                q0T1 += new_q0t1
                
                next_level.append(r.left)
                next_level.append(r.right)
        current_level = next_level
        cur_depth += 1
    return LT1, q0T1, root


"""
NOTE: This is VERY similar to ConstructT1. There is very likely a way to refactor this so that the code for T0 and 
T1 fit into one function, but I am too lazy to refactor this at this point. See comments in constructT1 for more 
details

This is the Dynamic programming algorithm for Constructing T0 from https://arxiv.org/pdf/2001.11170.pdf
Inputs:
    p: Probability distribution. List or numpy array
    C: The constant C from the paper

Returns:
    LT0: The expected codeword length of T0^*
    q1T0: The sum of master probabilities in T0^*
    root: A BstNode root of the tree T_1^* that offers .display() API to print it
"""
def constructT0(p, C):
    P = p
    precalc_Wm(p)
    n = len(p)
    I0 = [(0,2,0),(1,0,1)] #initial starting signatures
    Adj = {} #Adjacency matrix in Signature graph
    q = collections.deque(I0)
    
    while len(q)>0:
        s = q.popleft()
        if s not in Adj:
            Adj[s] = {}
            mp, pp, zp = s
            for e0 in range(pp+1):
                for e1 in range(pp+1):
                    if e0+e1<=pp:
                        m,p,z = mp+e0+e1, zp+2*(pp-e0-e1), e1
                        if (z>=0 and z<=m and m<=n and 0<=p and p<=n and 
                            (mp != m or pp!=p or zp!=z)):
                                next_signature = (m,p,z)
                                cost_expansion = WPrime(mp) + C*WTwo(mp-zp, mp)
                                Adj[s][next_signature] = cost_expansion
                                q.append(next_signature)
    
    all_sigs = Adj.keys()
    INF = 10*n #AIFV-2 code length at most 4*n, put 10*n for buffer, or just any big num
    
    ShortestDist = {sig : INF for sig in all_sigs} #Shortest dist from I0 to sig
    Pred = {sig : None for sig in all_sigs} #Preceding signature to sig in path
    PredExpand = {sig : None for sig in all_sigs} #Expansion from prev sig to reach here
    
    for i0 in I0:
        ShortestDist[i0] = 0
    
    topological_sort = sorted(list(all_sigs))
    for s in topological_sort:
        mp, pp, zp = s
        for e0 in range(pp+1):
            for e1 in range(pp+1):
                if e0+e1 <= pp:
                    m,p,z = mp+e0+e1, zp+2*(pp-e0-e1), e1
                    if (z>=0 and z<=m and m<=n and 0<=p and p<=n and 
                            (mp != m or pp!=p or zp!=z)):
                        next_sig = (m,p,z)
                        cost_expansion = WPrime(mp)+C*WTwo(mp-zp, mp)
                        total_cost = ShortestDist[s]+cost_expansion
                        if total_cost < ShortestDist[next_sig]:
                            ShortestDist[next_sig] = total_cost
                            Pred[next_sig] = s
                            PredExpand[next_sig] = (e0,e1)
    s = (n,0,0)
    signatures = [s]
    expansions = [PredExpand[s]]
    while Pred[s]:
        s = Pred[s]
        signatures = [s]+signatures
        if PredExpand[s]:
            expansions = [PredExpand[s]] + expansions
    
    LT0, q1T0 = 0,0
    if signatures[0] == (0,2,0):
        root = BstNode(0)
        root.complete = True
    elif signatures[0]==(1,0,1):
        root = BstNode(P[0])
        q1T0 += P[0]
        P = P[1:]
        root.master = True
    else:
        raise Exception("Something went terribly wrong with the starting signature")
    
    root.depth = 0
    
    cur_depth = 0
    current_level = [root]
    
    for a,b in expansions:
        leaves = a
        masters = b
        next_level = []
        Pis = P[: leaves+masters]
        
        idx_leaf = 0
        idx_mast = leaves
        P = P[leaves+masters:]
        
        for r in current_level:
            if r.leaf:
                continue
            
            if r.slave:
                if leaves > 0:
                    r.left = BstNode(Pis[idx_leaf])
                    LT0 += Pis[idx_leaf] * (cur_depth+1)
                    r.left.depth = cur_depth + 1
                    r.left.leaf = True
                    leaves = leaves - 1
                    idx_leaf = idx_leaf + 1
                elif masters > 0:
                    r.left = BstNode(Pis[idx_mast])
                    LT0 += Pis[idx_mast] * (cur_depth + 1)
                    r.left.depth = cur_depth + 1
                    r.left.master = True
                    masters = masters - 1
                    idx_mast = idx_mast - 1
                else:
                    r.left = BstNode(0)
                    r.left.depths = cur_depth + 1
                    r.left.complete = True
                next_level.append(r.left)
            
            if r.master:
                r.left = BstNode("xx")
                r.left.depth = cur_depth + 1
                r.left.slave = True
                next_level.append(r.left)
            
            if r.complete:
                r.left, leaves, masters, idx_leaf, idx_mast, lt0, new_q1t0, new_q0t1 = assign_child(leaves, masters, Pis, idx_leaf, idx_mast, cur_depth)
                LT0 += lt0
                q1T0 += new_q1t0
                
                r.right,leaves, masters, idx_leaf, idx_mast, lt0, new_q1t0, new_q0t1 = assign_child(leaves, masters, Pis, idx_leaf, idx_mast, cur_depth)  
                LT0 += lt0
                q1T0 += new_q1t0
                
                next_level.append(r.left)
                next_level.append(r.right)
        assert leaves == 0
        assert masters== 0
        current_level = next_level
        cur_depth += 1
    return LT0, q1T0, root


"""
This is Algorithm 2 from https://arxiv.org/pdf/2001.11170.pdf

p: The probability distribution as a list, or numpy array
b: Number of bits of probability distribution

Return: Nothing, The algorithm prints the Code tree in the end
"""
def binarySearch(p, b=5):
    l,r = 0,1
    epsilon = 2**(-2*(b+1))
    while r-l > epsilon:
        mid = (l+r)/2
        
        LT0, q1T0, root0 = constructT0(p, mid)
        LT1, q0T1, root1 = constructT1(p, mid)

        e0 = LT0 + mid*q1T0
        e1 = LT1 - mid*q0T1
        
        if e0<e1:
            l = mid
        else:
            r = mid
    print("Calculating AIFV Codes for %s"%( "[" + ','.join([str(x) for x in p]) + "]" ) )
    print("C^* = %s"%(mid))
    print("AIFV Code Cost: %s"%( (q0T1*LT0 + q1T0*LT1) / (q1T0 + q0T1)))
    print("Enrtopy: %s"%( sum( [-p[i]*log2(p[i]) for i in range(len(p))]) ))
    print("L(T_0^*) = %s, q_1(T_0^*) = %s"%(round(LT0, 2) ,round(q1T0, 2)))
    print("L(T_1^*) = %s, q_0(T_1^*) = %s"%(round(LT1, 2) ,round(q0T1, 2)))
    print("\nT_0^*")
    root0.display()
    print("\nT_1^*")
    root1.display()


#p = [0.5, 0.25, 0.2, 0.05]
#p = [0.9,0.02,0.02,0.02,0.02,0.02]


N = 15
np.random.seed(0)
p = np.random.random(N)
p /= p.sum()
p = p.round(2)
p = sorted(p, reverse=True)


binarySearch(p, b=5)

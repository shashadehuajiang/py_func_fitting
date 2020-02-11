# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 10:24:01 2019

@author: 90415
"""
import numpy as np
from types import FunctionType

foo_code = compile('def foo(a,b): return a+np.exp(b)', "<string>", "exec")
foo_func = FunctionType(foo_code.co_consts[0], globals(), "foo")
print(foo_func(1,2))
foo_code = compile('def foo(a,b): return a+b', "<string>", "exec")
foo_func = FunctionType(foo_code.co_consts[0], globals(), "foo")
print(foo_func(1,2))

import warnings
warnings.filterwarnings("ignore")

#%%
import time, timeit
def clock(func):
    def clocked(*args):
        t0 = timeit.default_timer()
        result = func(*args)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        #arg_str = ', '.join(repr(arg) for arg in args)
        print('[%0.8fs] %s' % (elapsed, name))
        return result
    return clocked

#%%
import numpy as np
from binarytree import Node
import re
import copy
import random


class BTree:
    def __init__(self,expression):
        if not self.Check_bracket(expression):
            print("Mismatched braces！！")
            print('expression:',expression)
            os.exit()
            return
        
        
        self.options =['(',')','[',']','{','}','sin','cos','log','**','*','/','+','-']
        self.single_ops = ['sin','cos','log']
        self.expression = expression
        self.elist = self.expression_to_list(expression)
        self.tree = self.buildtree(self.elist,[])
        #print('elist:',self.elist)
        #self.buildtree(self.elist)
    

    def get_paras(self):
        out = []
        unknown = 'abcdefghijklmnopqrstuvw'
        for i_e in self.elist:
            if len(i_e) == 1 and i_e in unknown:
                out.append(i_e)
        return out
        

    def Check_bracket(self,s):
        self.BRACKET = {'}': '{', ')': '(', ']': '['}
        self.BRACKET_L, self.BRACKET_R = self.BRACKET.values(), self.BRACKET.keys()
        arr = []
        for c in s:
            if c in self.BRACKET_L:
                # 左括号入栈
                arr.append(c)
            elif c in self.BRACKET_R:
                # 右括号，要么栈顶元素出栈，要么匹配失败
                if arr and arr[-1] == self.BRACKET[c]:
                    arr.pop()
                else:
                    return False
        if len(arr) ==0:
            return True
        return False
    
    
    def str_findall(self,str1,target):
        outlist = []
        now = 0
        while(now<len(str1)):
            p= str1[now:].find(target)
            if p>=0:
                outlist.append(p+now)
                now += len(target)+p
            else:
                break
        return outlist
        
    
    def expression_to_list(self,expression):
        options=copy.deepcopy(self.options)
        #print(expression)
        num = re.findall('\d+\.\d+|\d+',expression)
        num.sort(key=lambda ele:len(ele),reverse=True)
        #print('num',num)
        options = options + num
        #print('options',options)
        
        matched_list = np.zeros(len(expression))
        
        outlist = []
        id_number = 1
        
        for i_op in options:
            i_match_list = self.str_findall(expression,i_op)
            #print('i_match_list:',i_op,i_match_list)
            #print('matched_list',matched_list)
            for i_m in i_match_list:
                if matched_list[i_m] == 0:
                    matched_list[i_m:i_m+len(i_op)] = id_number
                    id_number +=1
        
        #print('matched_list',matched_list)
        
        startnumber = 0
        for i in range(len(matched_list)):
            if matched_list[i] == matched_list[startnumber]:
                continue
            else:
                outlist.append(expression[startnumber:i])
                startnumber = i
        outlist.append(expression[startnumber])
                
        return outlist
        
    
    
    def find_match_BRACKET_R(self,e_list,i):

        arr = []
        for ie in range(len(e_list)):
            if e_list[ie] in self.BRACKET_L:
                # 左括号入栈
                if ie == i:
                    arr.append([e_list[ie],1])
                else:
                    arr.append([e_list[ie],0])
            elif e_list[ie] in self.BRACKET_R:
                # 右括号，要么栈顶元素出栈，要么匹配失败
                if arr and arr[-1][0] == self.BRACKET[e_list[ie]]:
                    pout = arr.pop()
                    if pout[1] == 1:
                        return ie
                else:
                    return -1
        
        return 

    
    def buildtree(self,e_list_in,node_list = []):
        #print('e_list_in',e_list_in)#,'node_list',node_list)
        e_list = copy.deepcopy(e_list_in)
        #逐渐合并，由子节点合并为父节点
        if node_list == []:
            for i in range(len(e_list)):
                node1 = Node(i,None,None)
                node_list.append(node1)
        
        #print(node_list[0])
        
        if len(e_list)==1:
            #print('??????????????')
            #print(node_list,node_list[0])
            return node_list[0]
        

        
        options = copy.deepcopy(self.options[6:])
        
        #print(e_list[13],self.find_match_BRACKET_R(e_list,13))
        BRACKET_L = list(self.BRACKET_L)
        for i in range(len(BRACKET_L)):
            while BRACKET_L[i] in e_list:
                a = e_list.index(BRACKET_L[i])
                pair = self.find_match_BRACKET_R(e_list,a)
                
                
                sub_e_list = copy.deepcopy(e_list[a+1:pair])
                sub_node_list = copy.deepcopy(node_list[a+1:pair])
                
                new_node = self.buildtree(sub_e_list,sub_node_list)
                
                node_list[a] = new_node
                for j in range(a+1,pair+1):
                    e_list[a] += e_list[j]
                
                #合并子树
                for j in range(pair,a,-1):
                    del(e_list[j])
                    del(node_list[j])
                    
                #print(e_list)
                
            
        for i_op in options:
            while (i_op in e_list):
                a = e_list.index(i_op)
                
                start = 0
                end = 0
                if a>0:
                    #print('a' ,a,e_list[a-1] not in options and e_list[a+1] not in options)
                    if e_list[a-1] in options and e_list[a+1] not in options:
                        start = a
                        end = a+1
                    #双操作符
                    elif e_list[a-1] not in options and e_list[a+1] not in options:
                        start = a-1
                        end = a+1
                    else:
                        print('ERROR!!!')
                elif a == 0:
                    start = a
                    end = a+1
                
                #print('a' ,a,e_list)
                #如果这是个单操作符
                if end - start == 1:
                    node_list[a].left = Node(-1,None,None) 
                    #print(node_list)
                    node_list[a].right = node_list[a+1]
                    del(node_list[a+1])    
                    
                    e_list[a] += e_list[a+1]
                    del(e_list[a+1])    
                    
                #标准二叉
                elif end-start ==2:
                    #print(e_list[start:end+1])
                    node_list[a].left = node_list[a-1]
                    node_list[a].right = node_list[a+1]
                    #print(node_list[a])
                    del(node_list[a+1])
                    del(node_list[a-1])
                    #os.exit()
                    
                    e_list[a-1] += e_list[a]
                    e_list[a-1] += e_list[a+1]
                    del(e_list[a+1])    
                    del(e_list[a])  
               
                '''
                sub_e_list = copy.deepcopy(e_list[start:end+1])
                sub_node_list = copy.deepcopy(node_list[start:end+1])
                new_node = self.buildtree(sub_e_list,sub_node_list)
                '''

                
                
            #print(e_list)
             
        return node_list[0]
    
    def in_order_traversal(self,root):
        
        if not root.left == None:
            #print('T!')
            self.out_func_list.append('(')
            #print(root.left)
            self.out_func_list.append(self.in_order_traversal(root.left))#left
        #else:
        
            #print(root.left,'FFFF')
        #print(root.value)
        self.out_func_list.append(root.value)
        
        
        #此处给单项函数加上括号
        if not root.left == None:
            if root.left.value == -1:
                self.out_func_list.append('(')
        
        if not root.right == None:
            self.out_func_list.append(self.in_order_traversal(root.right))
            self.out_func_list.append(')')
        
         #此处给单项函数加上括号
        if not root.left == None:
            if root.left.value == -1:
                self.out_func_list.append(')')
        
        return
    
    def show_function(self):
        unknown = 'abcdefghijklmnopqrstuvw'
        for i_e in self.elist:
            if len(i_e) == 1:
                a = unknown.find(i_e)
                #print(a,i_e)
                if a >=0:
                    unknown = unknown[0:a] + unknown[a+1:]
        #print(self.elist)         
        
        startnum = 0
        outstr = ''
        self.out_func_list = []
        #print(self.tree.inorder)
        self.in_order_traversal(self.tree)
        #print(self.out_func_list)
        for i in self.out_func_list:
            #print(i)
            if not i == 0 and not i:
                continue
            if i in self.BRACKET_L:
                outstr += '('
                continue
            if i in self.BRACKET_R:
                outstr += ')'
                continue
            if i>=0:
                #print(i,self.elist)
                outstr += self.elist[i]
            elif i==-1:
                pass
            elif i == -2:  #随机参数
                outstr += unknown[startnum]
                startnum+=1
        return outstr
    
    def father(self,node1):
        nodes = self.tree.inorder
        for i_n in nodes:
            if i_n.left is node1 or i_n.right is node1:
                return i_n
    
    
    def tree_random_cut(self):
        #长度不够，什么都不做
        if len(self.tree)<=4:
            return
        
        #print('cut')
        
        #随机选择 child_node
        while True:
            nodes = self.tree.inorder
            i_r = random.randint(0,len(self.tree.inorder)-1)
            child_node = copy.deepcopy(nodes[i_r])
            if child_node.value is not -1 and child_node.left:
                break
        
        #print(child_node)
        #print('child_node',child_node)
        #可能不对
        operator = self.elist[child_node.value]
        #单操作符
        #print('operator',operator)
        if operator in self.single_ops:
            #print("in single..................")
            nodes[i_r].value = child_node.right.value
            nodes[i_r].left = child_node.right.left
            nodes[i_r].right = child_node.right.right
        else:#双操作符
            if random.random()>0.5:
                nodes[i_r].value = child_node.left.value
                nodes[i_r].left = child_node.left.left
                nodes[i_r].right = child_node.left.right
            else:
                nodes[i_r].value = child_node.right.value
                nodes[i_r].left = child_node.right.left
                nodes[i_r].right = child_node.right.right
        
        '''
        if i_r != 0:
            return
        
        elif i_r==0:
            return
        '''
        return
    
    
    def tree_random_add(self):
        #随机选择一个节点A，用A*操作符*变量B替代

        #随机选择 child_node
        while True:
            nodes = self.tree.inorder
            i_r = random.randint(0,len(self.tree.inorder)-1)
            child_node = copy.deepcopy(nodes[i_r])
            if child_node.value is not -1:
                break
                
        #print('i_r',i_r)
        #i_r = 16 
        #child_node = self.tree[i_r]#test
        
        if i_r != 0:
            #随机选择操作符
            options = copy.deepcopy(self.options[6:])
            i_o = random.randint(0,len(options)-1)
            operator = options[i_o]
            #operator = 'sin'
            #print('operator',operator)
            #node_root = Node(len(self.elist),None,None)
            #print('node_root',len(self.elist)+1)
            if operator in self.single_ops:
                #单操作符，另a=F（a）
                #node_root.left = Node(-1,None,None)
                #node_root.right = child_node
                nodes[i_r].right = child_node
                nodes[i_r].left = Node(-1,None,None)
                nodes[i_r].value = len(self.elist)
                #print('node_root',node_root)
                self.elist.append('')
                #print(len(self.tree),len(self.elist))
                self.elist[len(self.elist)-1] = operator
            else:
                #node_root = Node(len(self.tree))
                #随机参数
                foo = ['x', -2 ]#-2为待定参数
                random_a = random.choice(foo)
                if random_a == -2:
                    if random.random()>0.5:
                        nodes[i_r].left = Node(-2,None,None)
                        nodes[i_r].right = child_node
                    else:
                        nodes[i_r].left = child_node
                        nodes[i_r].right = Node(-2,None,None)
                    nodes[i_r].value = len(self.elist)
                    self.elist.append('')
                    self.elist[len(self.elist)-1] = operator
                
                elif random_a == 'x':
                    if random.random()>0.5:
                        nodes[i_r].left = Node(len(self.elist)+1,None,None)
                        nodes[i_r].right = child_node
                    else:
                        nodes[i_r].left = child_node
                        nodes[i_r].right = Node(len(self.elist)+1,None,None)
                    nodes[i_r].value = len(self.elist)
                    self.elist.append('')
                    self.elist.append('')
                    self.elist[len(self.elist)-2] = operator
                    self.elist[len(self.elist)-1] = random_a
                    
        #为根节点            
        elif i_r==0:
            #随机选择操作符
            options = copy.deepcopy(self.options[6:])
            i_o = random.randint(0,len(options)-1)
            operator = options[i_o]
            #print('operator',operator)
            #node_root = Node(len(self.elist),None,None)
            #print('node_root',len(self.elist)+1)
            if operator in self.single_ops:
                #单操作符，另a=F（a）
                nodes[i_r].left = Node(-1,None,None)
                nodes[i_r].right = child_node
                nodes[i_r].value = len(self.elist)
                self.elist.append('')
                #print(len(self.tree),len(self.elist))
                self.elist[len(self.elist)-1] = operator
            else:
                #node_root = Node(len(self.tree))
                #随机参数
                foo = ['x', -2 ]#-2为待定参数
                random_a = random.choice(foo)
                if random_a == -2:
                    if random.random()>0.5:
                        nodes[i_r].left = Node(-2,None,None)
                        nodes[i_r].right = child_node
                    else:
                        nodes[i_r].left = child_node
                        nodes[i_r].right = Node(-2,None,None)
                    nodes[i_r].value = len(self.elist)
                    self.elist.append('')
                    self.elist[len(self.elist)-1] = operator
                
                elif random_a == 'x':
                    if random.random()>0.5:
                        nodes[i_r].left = Node(len(self.elist)+1,None,None)
                        nodes[i_r].right = child_node
                    else:
                        nodes[i_r].left = child_node
                        nodes[i_r].right = Node(len(self.elist)+1,None,None)
                    nodes[i_r].value = len(self.elist)
                    self.elist.append('')
                    self.elist.append('')
                    self.elist[len(self.elist)-2] = operator
                    self.elist[len(self.elist)-1] = random_a
                #print('node_root',node_root)
            #os.exit()
            

#bt1 = BTree('a**3+2*b+c*99.1+sin(x+1+222)')
bt1 = BTree('(((x+d)**(cos(((c+x)*b))))+a)')
print(bt1.tree)
print(bt1.show_function())
print(bt1.elist)

#bt1.tree_random_add()
bt1.tree_random_cut()
print(bt1.tree)
print(bt1.show_function())

print(len(bt1.tree))

#%%
#优化过程
#import math

def softmax(x):
    x -= x.max()
    x = np.exp(x)
    return x / x.sum()

def takeSecond(elem):
    return elem[1]

class Evolution_f:
    def __init__(self,xdata,ydata,function_number=100,episode = 100,alpha = 0.1,hintfunction=''):
        self.xdata = xdata
        self.ydata = ydata
        self.function_number = function_number
        self.hintfunction = hintfunction
        self.episode = episode
        self.alpha=alpha
        
        self.functions = []
        #self.test_func(self.hintfunction)
        
        #return self.start()
        
    
    def ini(self):

        for i in range(self.function_number):
            rand_f = self.generate_function('x')
            self.functions.append([rand_f,999])
            #print(rand_f)
            
        if self.hintfunction:
            self.functions[0][0] = self.hintfunction
        
        self.refresh()
        return
    
    def refresh(self):
        for i_f_v in range(len(self.functions)):
            function = self.functions[i_f_v][0]
            self.functions[i_f_v][1] = self.test_func(function)
            #print(function,self.functions[i_f_v][1])
            #os.exit()
        
    def random_index(self,p1):
        rand_s = random.random()
        sum1 = 0
        for i in range(len(p1)):
            sum1 += p1[i]
            if sum1>= rand_s:
                return i
        
    
    def evolution_step(self):
        #排序
        self.functions.sort(key=takeSecond)
        
        
        #按照softmax（1-value）的概率进行生成
        values = np.zeros(len(self.functions))
        for i_fv in range(len(self.functions)):
            value1 = 1/self.functions[i_fv][1]
            values[i_fv] = value1
        values = softmax(values)
        
        #生成20%新的增长
        newfunctions = []
        numberin = 0.2*self.function_number
        for i in range(int(numberin)):
            random_i = self.random_index(values)
            oldfunction = self.functions[random_i][0]
            bt1 = BTree(oldfunction)
            bt1.tree_random_add()
            newf = bt1.show_function()
            #test
            try:
                bt1 = BTree(newf)
            except:
                print('oldfunction',oldfunction)
                print('newfunction',newf)
            newfunctions.append(newf)
        
        #生成20%新的剪枝
        for i in range(int(numberin)):
            random_i = self.random_index(values)
            oldfunction = self.functions[random_i][0]
            bt1 = BTree(oldfunction)
            bt1.tree_random_cut()
            newf = bt1.show_function()
            #test
            try:
                bt1 = BTree(newf)
            except:
                print('oldfunction',oldfunction)
                print('newfunction',newf)
            newfunctions.append(newf)
        
        #用新的方程替代末尾的
        for i in range(len(newfunctions)):
            newf = newfunctions[i]
            self.functions[len(self.functions)-i-1][0] = newf
            self.functions[len(self.functions)-i-1][1] = self.test_func(newf)
            
        return  
    
    def start(self):
        self.ini()
        for i in range(self.episode):    
            self.evolution_step()
            print(self.functions[0],self.functions[1])
            
        return self.functions[0][0]
        
    
    def generate_function(self,func_in):
        bt2 = BTree(func_in)
        #print('build,finish...')
        bt2.tree_random_add()
        #print('rand,finish...')
        return bt2.show_function()


    def get_paras(self,function):
        bt1 = BTree(function)
        return bt1.get_paras()
    
    
    def use_np_functions(self,function):
        text = function.replace('sin','np.sin')
        text = text.replace('cos','np.cos')
        text = text.replace('log','np.log')
        return text
    
    def delete_np(self,function):
        a = function.find('np.')
        while (a>=0):
            function = function[0:a] + function[a+3:]
            a = function.find('np.')
        
    def test_func(self,function):
        param_s = self.get_paras(function)
        #print(param_s)
        
        
        bt1 = BTree(function)
        information = len(bt1.tree)
        msecost = 0
        
        
        func_text = 'def foo(x'
        for i_p in param_s:
            func_text += ','
            func_text += i_p
        func_text += '): return '
        func_text += function
        func_text = self.use_np_functions(func_text)
        #print('func_text',func_text)

        if len(param_s) >0:
            #foo_code = compile('def foo(a,b): return a+np.exp(b)', "<string>", "exec")
            foo_code = compile(func_text, "<string>", "exec")
            foo_func = FunctionType(foo_code.co_consts[0], globals(), "foo")
            try:
                popt,msecost = curve_fit(foo_func, self.xdata, self.ydata,my_return = True)
                #print(func_text,'popt',popt,'cost',msecost)
            except:
                msecost = 9999999
        else:
            msecost = 9999999
        #self.alpha = 0.1
        Error = self.alpha*information + (1-self.alpha)*msecost
        
        return Error
        

#%%
#测试过程
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


@clock
def test():
    x = np.linspace(0, 1, 100)
    #y = x
    y = 10*np.log(x+0.1)+2.3
    y_data = y + np.random.normal(loc=0,scale=0.05,size=100)
    # '.'标明画散点图，每个散点的形状是个圆
    plt.plot(x, y_data, '.')
     
    # 画模型的图，plot函数默认画连线图
    #plt.figure('model')
    plt.plot(x, y)
    
    #return
    #xdata,ydata,function_number=100,episode = 100,alpha = 0.1,hintfunction=''
    ev1 = Evolution_f(x,y,2000,30,0.1,'')
    outfunc = ev1.start()
    
    return outfunc

func = test()
print(func)









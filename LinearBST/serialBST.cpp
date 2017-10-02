/*
 ***
 Question Name:
 ***
 Question Link:
 ***
 Idea:
 */

#include <memory.h>
#include <ctime>
#include <random>
#include <iomanip>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <fstream>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cstdio>
#include <vector>
#include <cmath>
#include <functional>
#include <queue>
#include <deque>
#include <stack>
#include <list>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>

#define REP(i,s,n) for(int (i)=s; (i)<(int)(n);(i)++)
#define RIT(it,c) for(__typeof(c.begin()) it = c.begin();it!=c.end();it++)
#define ALL(x) x.begin(), x.end()
#define SZ(x) (int)(x).size()
#define MSET(m,v) memset(m,v,sizeof(m))

using namespace std;

typedef long double ld;
typedef vector<int> vi;
typedef vector<long> vl;
typedef vector<bool> vb;
typedef vector<double> vd;
typedef pair<int,int> ii;
typedef pair<long, long> ll;
typedef unordered_set<int> ui;

const int MAX_SIZE = 1024;
const double error = 1.e-8;
double keys[MAX_SIZE];
double values[MAX_SIZE];
int children[MAX_SIZE];
int parent[MAX_SIZE];
int rest_idx[MAX_SIZE];
int tree_root, rest_room = MAX_SIZE;


int compare(double x, double y){
    if(x <= y-error) return -1;
    if(x >= y+error) return 1;
    return 0;
}

/*
 
 
 
 */

void initTree(){
    rest_room = MAX_SIZE-1;
    for(int i=0;i<MAX_SIZE;++i) rest_idx[i] = MAX_SIZE - i - 1;
    memset(parent, 0, sizeof(parent));
    memset(children, 0 , sizeof(children));
    memset(keys, 0, sizeof(keys));
    memset(values, 0, sizeof(values));
}

bool isEmpty(){
    return rest_room == MAX_SIZE-1;
}

bool isFull(){
    return !rest_room;
}

int getSize(){
    return MAX_SIZE - rest_room - 1;
}

bool searchTree(double key){
    if(rest_room == MAX_SIZE) return false;
    int root = tree_root, judge;
    while((judge = compare(key, keys[root]))){
        if(judge == -1){
            if(!(children[root]/MAX_SIZE)) return false;
            root = children[root]/MAX_SIZE;
        }
        else{
            if(!(children[root]%MAX_SIZE)) return false;
            root = children[root]%MAX_SIZE;
        }
    }
    return true;
}

double getItem(double key){
    assert(!isEmpty());
    int root = tree_root, judge;
    while((judge = compare(key, keys[root]))){
        if(judge == -1){
            assert(children[root]/MAX_SIZE);
            root = children[root]/MAX_SIZE;
        }
        else{
            assert(children[root]%MAX_SIZE);
            root = children[root]%MAX_SIZE;
        }
    }
    return values[root];
}

void insertTree(double key, double value){
    assert(rest_room);
    if(isEmpty()){
        --rest_room;
        tree_root = rest_idx[rest_room];
        keys[tree_root] = key;
        values[tree_root] = value;
        parent[tree_root] = 0;
        children[0] = tree_root;
        return;
    }
    int root = tree_root, judge;
    while((judge = compare(key, keys[root]))){
        if(judge == -1){
            if(!(children[root]/MAX_SIZE)){
                --rest_room;
                int idx = rest_idx[rest_room];
                keys[idx] = key;
                values[idx] = value;
                parent[idx] = root;
                children[root] += idx*MAX_SIZE;
                return;
            }
            root = children[root]/MAX_SIZE;
        }
        else{
            if(!(children[root]%MAX_SIZE)){
                --rest_room;
                int idx = rest_idx[rest_room];
                keys[idx] = key;
                values[idx] = value;
                parent[idx] = root;
                children[root] += idx;
                return;
            }
            root = children[root]%MAX_SIZE;
        }
    }
    values[root] = value;
    return;
}

void eraseTree(double key){
    if(isEmpty()) return;
    int root = tree_root, judge;
    while((judge = compare(key, keys[root]))){
        if(judge == -1){
            if(!(children[root]/MAX_SIZE)) return;
            root = children[root]/MAX_SIZE;
        }
        else{
            if(!(children[root]%MAX_SIZE)) return;
            root = children[root]%MAX_SIZE;
        }
    }
    int par = parent[root];
    if(!(children[root]/MAX_SIZE)){
        if(children[par]/MAX_SIZE == root){
            children[par] = children[par]%MAX_SIZE + (children[root]%MAX_SIZE)*MAX_SIZE;
        }
        else {
            children[par] += children[root]%MAX_SIZE - root;
        }
        if(children[root]) parent[children[root]] = par;
    }
    else if(!(children[children[root]/MAX_SIZE] % MAX_SIZE)){
        if(children[par]/MAX_SIZE == root){
            children[par] = children[par]%MAX_SIZE + (children[root]/MAX_SIZE)*MAX_SIZE;
        }
        else {
            children[par] += children[root]/MAX_SIZE - root;
        }
        children[children[root]/MAX_SIZE] += children[root]%MAX_SIZE;
        if(children[root]%MAX_SIZE) parent[children[root]%MAX_SIZE] = children[root]/MAX_SIZE;
    }
    else{
        int current = children[children[root]/MAX_SIZE] % MAX_SIZE;
        while(children[current]%MAX_SIZE) current = children[current]%MAX_SIZE;
        children[parent[current]] -= current;
        children[current]= children[root];
        parent[children[root]/MAX_SIZE] = current;
        if(children[root]%MAX_SIZE) parent[children[root]%MAX_SIZE] = current;
        if(children[par]/MAX_SIZE == root){
            children[par] = children[par]%MAX_SIZE + current*MAX_SIZE;
        }
        else {
            children[par] += current - root;
        }
        parent[current] = par;
    }
    parent[root] = children[root] = 0;
    keys[root] = values[root] = 0.;
    rest_idx[rest_room] = root;
    ++rest_room;
    return;
}





int main(){
    std::ios_base::sync_with_stdio(false),cin.tie(0),cout.tie(0);
    initTree();
    cout<<"Empty? :  "<<isEmpty()<<endl;
    cout<<"Full? :  "<<isFull()<<endl;
    double k[10] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    double v[10] = {1.23, 2.12, 3.56, 4.12, 5.23, 6.78, 7.85, 8.32, 9.32, 10.34};
    for(int i=0;i<10;++i) insertTree(k[i], v[i]);
    cout<<getSize()<<endl;
    cout<<"5 in the tree? :  "<<searchTree(5.)<<endl;
    cout<<"Get the value of 5:  "<<getItem(5.)<<endl;
    cout<<"5.5 in the tree? :  "<<searchTree(5.5)<<endl;
    eraseTree(5.);
    cout<<getSize()<<endl;
    cout<<"5 in the tree? :  "<<searchTree(5.)<<endl;
    cout<<endl;
    return 0;
}


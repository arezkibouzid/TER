#include "cDisjointSet.h"
#include<iostream>


cDisjointSet::cDisjointSet() : elements(0), sets(0) {
}

cDisjointSet::~cDisjointSet() {
    for (int i = 0; i < nodes.size(); i++) delete nodes[i];
    nodes.clear();
}



node* cDisjointSet::MakeSet(int i) {

    int t = elements + 1;
    if ( t> nodes.size()) nodes.push_back(new node);
    nodes[elements]->parent = nodes[elements];
    nodes[elements]->i = i;
    nodes[elements]->rank = 0;
    elements++;
    sets++;
    t = elements-1;
    return nodes[t];

}

node* cDisjointSet::Find(node* a) {

    if (a->parent == a) return a;
    else
    {
        a->parent = Find(a->parent);
        return a->parent;

    }

}

void cDisjointSet::Union(node* a0, node* a1) {

    if (a0 == a1) return;

    node* a2 = Find(a0);
    node* a3 = Find(a1);

    if (a2 == a3) return;

    if (a2->rank < a3->rank) { a2->parent = a3;  sets--; }
    else if (a3->rank < a2->rank){ a3->parent = a2;  sets--;}
    /*else {
        a3->parent = a2;
        a2->rank++;
        sets--;
    }*/

   

}

int cDisjointSet::ElementCount() {
    return elements;
}

int cDisjointSet::SetCount() {
    return sets;
}

int cDisjointSet::Reduce() {
    int j = 0;
    for (int i = 0; i < elements; i++)
        if (nodes[i]->parent == nodes[i])
            nodes[i]->i = j++;
    return j;
}

void cDisjointSet::Reset() {
    this->nodes.clear();
    elements = sets = 0;
}

void cDisjointSet::affiche(){
        std::string s = "";
        node* L;
        for (int i = 0; i < elements; i++)
        {
            L = nodes[i];
            while (L->parent == L) {
                std::cout << nodes[i]->rank << "|";
                std::cout << nodes[i]->i << "|";

                L= L->parent;

            }
            std::cout << std::endl;
        }
        
}
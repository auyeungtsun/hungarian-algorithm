#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cassert>

using namespace std;

const int INF = numeric_limits<int>::max();

class Hungarian {
private:
    int n;                     // The size of the cost matrix (NxN).
    vector<vector<int>> cost; // The cost matrix representing the bipartite graph.
    vector<int> u;             // Potential values for the first set of nodes.
    vector<int> v;             // Potential values for the second set of nodes.
    vector<int> p;             // Matching array: p[j] = i means that j is matched with i.
    vector<int> way;           // Array used to reconstruct the path during the algorithm.

public:
    /**
     * @brief Constructor for the Hungarian class.
     * @param costMatrix The cost matrix representing the bipartite graph.
     *                   The dimensions should be NxN.
     */
    Hungarian(const vector<vector<int>>& costMatrix) :
        n(costMatrix.size()),
        cost(costMatrix),
        u(n + 1, 0), v(n + 1, 0), p(n + 1, 0), way(n + 1, 0) {}

    /**
     * @brief Solves the minimum weight matching problem using the Hungarian algorithm.
     * 
     * This function computes the minimum weight matching for a bipartite graph 
     * represented by the cost matrix. It modifies the input `matching` vector 
     * to store the matching result, where matching[i] = j means that the i-th 
     * element in the first set is matched with the j-th element in the second set.
     * 
     * The potentials must always satisfy u[i] + v[j] <= cost[i][j] for all i, j.
     * At any stage of the algorithm, for any matched pair (i, j), cost[i][j] = u[i] + v[j].
     * Stores the minimum slack found so far for reaching column j.
     * minv[j] = min(cost[i0][j] - u[i0] - v[j]), where i0 is in the tree and j is outside.
     * The slack for an edge (i0, j) becomes cost[i0][j] - (u[i0] + delta) - v[j].
     * The minimum slack edge now has slack delta - delta = 0, bringing it into the equality subgraph.
     * For columns j in the tree, increase u[p[j]] by delta and decrease v[j] by delta.
     * These edges stay in the equality subgraph.
     * 
     * @param matching A vector to store the matching result.
     * @return The minimum total cost of the matching.
     * 
     * @note
     * Time Complexity: O(n^3)
     * Space Complexity: O(n^2)
     */
    int solve(vector<int>& matching) {
        for (int i = 1; i <= n; ++i) {
            p[0] = i;
            vector<int> minv(n + 1, INF);  // minv[j] stores the minimum slack value for node j.
            vector<bool> used(n + 1, false); // used[j] indicates if node j has been visited in the current path.
            int j0 = 0;

            do {
                used[j0] = true;
                int i0 = p[j0], delta = INF, j1 = 0;
                for (int j = 1; j <= n; ++j) {
                    if (!used[j]) {
                        int cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                        if (cur < minv[j]) {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                        if (minv[j] < delta) {
                            delta = minv[j];
                            j1 = j;
                        }
                    }
                }
                for (int j = 0; j <= n; ++j) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }
                j0 = j1;
            } while (p[j0] != 0);

            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0);
        }

        matching.resize(n);
        for (int j = 1; j <= n; ++j) {
            matching[p[j] - 1] = j - 1;
        }
        return -v[0];
    }
};

void test_hungarian() {
    cout << "Running Hungarian Algorithm Tests..." << endl;

    // Test Case 1: 1x1 Matrix
    {
        cout << "  Test Case 1: 1x1 Matrix..." << flush;
        vector<vector<int>> cost = {{5}};
        Hungarian solver(cost);
        vector<int> matching;
        int min_cost = solver.solve(matching);

        assert(min_cost == 5);
        vector<int> expected_matching = {0};
        assert(matching == expected_matching);
        cout << " Passed." << endl;
    }

    // Test Case 2: 2x2 Simple Diagonal
    {
        cout << "  Test Case 2: 2x2 Simple Diagonal..." << flush;
        vector<vector<int>> cost = {{1, 10}, {10, 2}};
        Hungarian solver(cost);
        vector<int> matching;
        int min_cost = solver.solve(matching);

        assert(min_cost == 3);
        vector<int> expected_matching = {0, 1};
        assert(matching == expected_matching);
        cout << " Passed." << endl;
    }

    // Test Case 3: 2x2 Simple Anti-Diagonal
    {
         cout << "  Test Case 3: 2x2 Simple Anti-Diagonal..." << flush;
        vector<vector<int>> cost = {{10, 1}, {2, 10}};
        Hungarian solver(cost);
        vector<int> matching;
        int min_cost = solver.solve(matching);

        assert(min_cost == 3);
        vector<int> expected_matching = {1, 0};
        assert(matching == expected_matching);
         cout << " Passed." << endl;
    }

    // Test Case 4: 3x3 Standard Example
    {
         cout << "  Test Case 4: 3x3 Standard Example..." << flush;
        vector<vector<int>> cost = {{8, 7, 9}, {4, 6, 3}, {5, 2, 8}};
        Hungarian solver(cost);
        vector<int> matching;
        int min_cost = solver.solve(matching);

        assert(min_cost == 13);
        vector<int> expected_matching = {0, 2, 1};
        assert(matching == expected_matching);
         cout << " Passed." << endl;
    }

    // Test Case 5: 3x3 With Zeros (Optimal cost is 0)
    {
         cout << "  Test Case 5: 3x3 With Zeros..." << flush;
        vector<vector<int>> cost = {{1, 2, 0}, {2, 0, 1}, {0, 1, 2}};
        Hungarian solver(cost);
        vector<int> matching;
        int min_cost = solver.solve(matching);

        assert(min_cost == 0);

        int calculated_cost = 0;
        vector<bool> job_assigned(cost.size(), false);
        assert(matching.size() == cost.size());
        vector<int> expected_matching = {2, 1, 0};
        assert(matching == expected_matching);

         cout << " Passed." << endl;
    }

     // Test Case 6: 4x4 Example
    {
         cout << "  Test Case 6: 4x4 Example..." << flush;
        vector<vector<int>> cost = {{90, 75, 75, 80},
                                    {35, 85, 55, 65},
                                    {125, 95, 90, 105},
                                    {45, 110, 95, 115}};
        Hungarian solver(cost);
        vector<int> matching;
        int min_cost = solver.solve(matching);

        assert(min_cost == 275);
        vector<int> expected_matching = {1, 3, 2, 0};
        assert(matching == expected_matching);

        cout << " Passed." << endl;
    }

    // Test Case 7: 5x5 General Matrix
    {
        cout << "  Test Case 7: 5x5 General Matrix..." << flush;
        vector<vector<int>> cost = {{15, 30, 10, 25, 20},
                                    {20, 10, 25, 15, 30},
                                    {25, 20, 15, 30, 10},
                                    {10, 25, 30, 20, 15},
                                    {30, 15, 20, 10, 25}};
        Hungarian solver(cost);
        vector<int> matching;
        int min_cost = solver.solve(matching);

        assert(min_cost == 50);

        vector<int> expected_matching = {2, 1, 4, 0, 3};
        assert(matching == expected_matching);
        cout << " Passed." << endl;
    }

    // Test Case 8: All Equal Costs (Degenerate Case)
    {
        cout << "  Test Case 8: 4x4 All Equal Costs..." << flush;
        vector<vector<int>> cost = {{7, 7, 7, 7},
                                    {7, 7, 7, 7},
                                    {7, 7, 7, 7},
                                    {7, 7, 7, 7}};
        Hungarian solver(cost);
        vector<int> matching;
        int min_cost = solver.solve(matching);

        assert(min_cost == 28);
        assert(matching.size() == 4);

        vector<bool> matched_values(4, false);
        for(int val : matching){
            assert(val >= 0 && val <= 3);
            matched_values[val] = true;
        }
        for (bool matched : matched_values) assert(matched);

        cout << " Passed." << endl;
    }


    cout << "All Hungarian Algorithm tests passed!" << endl;
}

void run_hungarian_sample() {

    cout << "\nExample Usage:" << endl;
    vector<vector<int>> production_costs = {{10, 12, 19, 11},
                                           { 5, 10,  7,  8},
                                           {12, 14, 13, 11},
                                           { 8, 15, 11,  9}};
    Hungarian prod_solver(production_costs);
    vector<int> prod_matching;
    int min_prod_cost = prod_solver.solve(prod_matching);

    cout << "Minimum Production Cost: " << min_prod_cost << endl;
    cout << "Matching (Machine -> Job):" << endl;
    for(int i=0; i < prod_matching.size(); ++i) {
        cout << "  Machine " << i << " -> Job " << prod_matching[i]
             << " (Cost: " << production_costs[i][prod_matching[i]] << ")" << endl;
    }

}

int main() {
    test_hungarian();
    run_hungarian_sample();
    return 0;
}
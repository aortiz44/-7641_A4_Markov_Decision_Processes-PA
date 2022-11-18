import mdptoolbox.example
import ForestHelper as FH

SUB_DIRECTORY = ["QL_EpsilonState/", "QL_ForestManagement/", "mdpPI/", "mdpVI/", "mdpQL/"]


def main_call():
    run_PI = True
    run_VI = True
    run_QL = True
    run_trianQL = False
    run_run_forest = True

    states = 2000
    # states = 255
    prob = 0.1
    P, R = mdptoolbox.example.forest(S=states, r1=4, r2=2, p=prob)
    state_range = range(4, states, 4)
    state_range = [states]
    if run_PI:
        print("Policy Iteration")
        FH.runPI_VI(FH.mdpPI, "Policy Iteration " + str(states) + " States ", state_range, SUB_DIRECTORY[2])
    if run_VI:
        print("Value Iteration")
        FH.runPI_VI(FH.mdpVI, "Value Iteration " + str(states) + " States ", state_range, SUB_DIRECTORY[3])
    if run_QL:
        print("Value Iteration on QLearning")
        FH.runPI_VI(FH.mdpQL, "Value Iteration on QLearning " + str(states) + " States ", state_range, SUB_DIRECTORY[4])
    if run_trianQL:
        print("Value Iteration on QLearning")
        FH.trainQL(P, R, states)
    if run_run_forest:
        print("STARTING FOREST MANAGEMENT ", str(states), " STATES")
        FH.run_Forest(states, P, R)  # States, test_count
    # FH.runPI()
    # FH.runVI()

    pass


main_call()

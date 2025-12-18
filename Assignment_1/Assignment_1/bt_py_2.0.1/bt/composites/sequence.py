import bt_library as btl

class Sequence(btl.Composite):
    """
    Specific implementation of the sequence composite.
    Runs children in order until one fails or runs.
    """

    def __init__(self, children: btl.NodeListType):
        """
        Constructor for the Sequence composite.

        :param children: List of child nodes
        """
        super().__init__(children)

    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        """
        Executes children one by one in sequence.

        :param blackboard: The blackboard state
        :return: SUCCEEDED if all children succeed, FAILED if any fail, RUNNING if any are still running
        """
        running_child = self.additional_information(blackboard, 0)

        for child_position in range(running_child, len(self.children)):
            child = self.children[child_position]

            result_child = child.run(blackboard)

            if result_child == btl.ResultEnum.FAILED:
                return self.report_failed(blackboard, 0)

            if result_child == btl.ResultEnum.RUNNING:
                return self.report_running(blackboard, child_position)

        return self.report_succeeded(blackboard, 0)

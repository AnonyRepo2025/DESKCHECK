"""Plan Monitor - Independent phase tracking for agent actions."""

__version__ = "0.1.0"

# Export main interfaces for easy integration
from simagent.plan_monitor.monitor import StatefulPhaseMonitor
from simagent.plan_monitor.phases import ActionEvent, MonitorResult

__all__ = ["StatefulPhaseMonitor", "ActionEvent", "MonitorResult", "__version__"]

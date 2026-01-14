FUNCTION_BLOCK FB_OvenZone
VAR_INPUT
    Enable          : BOOL := TRUE;
    Setpoint        : REAL := 85.0;
    
    // COUPLING INPUT: Represents the valve position of the upstream radiator
    // For Zone 1, this will be 0.
    // For Zone 2, this will be Zone 1's Valve Position.
    Coupling_In     : REAL := 0.0; 
    
    // How much does the upstream valve hurt me?
    // 0.41 means if upstream is 100% open, I lose 41% of my power.
    Coupling_Ratio  : REAL := 0.41; 
END_VAR
VAR_OUTPUT
    Valve_Pos       : REAL;
    Slave_SP        : REAL;
	Oven_Temp       : REAL;
END_VAR
VAR
    // Controllers
    MasterPID       : PID;
    SlavePID        : PID;
    
    // Physics State
    Raw_Air_Temp    : REAL;
    Dtime           : REAL := 2.0; // Task
    SimSpeed        : REAL := 10.0;
    
    // Helpers
    Effective_Valve : REAL;
    Max_Change      : REAL;
    T2_Seconds      : REAL;
    Actual_K_Gain   : REAL := 0.8125;
    Tau_2           : TIME := T#1169s;
END_VAR

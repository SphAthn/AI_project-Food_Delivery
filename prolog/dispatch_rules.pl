% SYMBOLIC PROLOG DISPATCH RULE ENGINE FOR FOOD DELIVERY ETA ADAPTATION
% File: prolog/dispatch_rules.pl
% Description: Evaluates operational safety guardrails, weather/traffic penalties,
%              and computes final hybrid ETA adjustments over ML baselines.


:- module(dispatch_rules, [
    evaluate_order/8,
    evaluate_batch/2,
    process_json_batch/2,
    weather_penalty/2,
    traffic_penalty/2,
    festival_penalty/2,
    multi_delivery_penalty/2,
    safety_buffer/1
]).

:- use_module(library(http/json)).


% 1. OPERATIONAL DISPATCH GUARDRAILS (FLAGGING REJECTIONS & HIGH-RISK ORDERS)

% Rule 1A: Reject order if distance exceeds absolute safety threshold (> 25 km)
dispatch_flag(Distance, _, _, _, _, 'REJECT_DISTANCE_EXCEEDED') :-
    Distance > 25.0, !.

% Rule 1B: High risk if low rating (< 3.8) during Jam/High traffic conditions
dispatch_flag(_, Rating, Traffic, _, _, 'RISK_LOW_RATING_PEAK_TRAFFIC') :-
    Rating < 3.8,
    (Traffic == 'Jam' ; Traffic == 'High'), !.

% Rule 1C: High risk if multiple deliveries (> 1.0) during severe weather
dispatch_flag(_, _, _, Weather, MultiDeliveries, 'RISK_MULTIPLE_DELIVERIES_BAD_WEATHER') :-
    MultiDeliveries > 1.0,
    (Weather == 'Stormy' ; Weather == 'Sandstorms' ; Weather == 'Fog'), !.

% Default: Order approved for standard dispatch
dispatch_flag(_, _, _, _, _, 'APPROVED').

% STANDARD SAFETY BUFFER ( IN MINUTES)
% Calibrated with penalty_calibration.py: raw ML ETA is late 48.1% of the time. A +2 min buffer lowers the late rate to 28.8% for a +0.46 min MAE cost.
safety_buffer(2.0).

% 2. WEATHER PENALTY RULES (MINUTES)

weather_penalty('Stormy', 1.0) :- !.
weather_penalty('Sandstorms', 1.0) :- !.
weather_penalty('Fog', 1.0) :- !.
weather_penalty(_, 0.0).

% 3. TRAFFIC PENALTY RULES (MINUTES)

traffic_penalty('Jam', 1.0) :- !.
traffic_penalty('High', 1.0) :- !.
traffic_penalty(_, 0.0).


% 4. FESTIVAL SURGE PENALTY RULES (MINUTES)

festival_penalty(_, 0.0).


% 5. MULTIPLE DELIVERIES LOAD PENALTY RULES (MINUTES)
% Penalty per extra delivery (minutes)
multi_delivery_unit_penalty(0.5).

multi_delivery_penalty(NumDeliveries, Penalty) :-
    NumDeliveries > 0,
    multi_delivery_unit_penalty(Unit),
    Penalty is NumDeliveries * Unit, !.
multi_delivery_penalty(_, 0.0).


% 6. MAIN HYBRID EVALUATION INFERENCE PREDICATE

evaluate_order(MlEta, Distance, Rating, Weather, Traffic, Festival, MultiDeliveries, [FinalHybridEta, TotalPrologPenalty, DispatchFlag]) :-

    % 1. Determine Dispatch Guardrail Flag
    dispatch_flag(Distance, Rating, Traffic, Weather, MultiDeliveries, DispatchFlag),

    % 2. Calculate Individual Symbolic Adjustments
    weather_penalty(Weather, WPen),
    traffic_penalty(Traffic, TPen),
    festival_penalty(Festival, FPen),
    multi_delivery_penalty(MultiDeliveries, MPen),
    safety_buffer(Buffer),

    % 3. Aggregate Total Symbolic Penalty
    TotalPrologPenalty is Buffer + WPen + TPen + FPen + MPen,

    % 4. Compute Final Hybrid ETA
    FinalHybridEta is MlEta + TotalPrologPenalty.

% 7. BATCH EVALUATION PREDICATES (HIGH-SPEED JSON FILE IO)
evaluate_batch([], []).
evaluate_batch([[MlEta, Distance, Rating, Weather, Traffic, Festival, MultiDel]|RestInputs],
               [[FinalEta, Penalty, Flag]|RestOutputs]) :-
    evaluate_order(MlEta, Distance, Rating, Weather, Traffic, Festival, MultiDel, [FinalEta, Penalty, Flag]),
    evaluate_batch(RestInputs, RestOutputs).

process_json_batch(InputFile, OutputFile) :-
    open(InputFile, read, InStream),
    json_read_dict(InStream, Dict),
    close(InStream),
    evaluate_batch(Dict.data, Results),
    open(OutputFile, write, OutStream),
    json_write(OutStream, Results),
    close(OutStream).

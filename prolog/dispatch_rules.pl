% SYMBOLIC PROLOG DISPATCH RULE ENGINE FOR FOOD DELIVERY ETA ADAPTATION
% File: prolog/dispatch_rules.pl
% Description: Evaluates operational safety guardrails, weather/traffic penalties,
%              and computes final hybrid ETA adjustments over ML baselines.

:- module(dispatch_rules, [
    evaluate_order/8
]).

% 1. OPERATIONAL DISPATCH GUARDRAILS (FLAGGING REJECTIONS & HIGH-RISK ORDERS)

% Rule 1A: Reject order if distance exceeds absolute safety threshold (> 25 km)
dispatch_flag(Distance, _, _, _, 'REJECT_DISTANCE_EXCEEDED') :-
    Distance > 25.0, !.

% Rule 1B: High risk if low rating (< 3.8) during Jam/High traffic conditions
dispatch_flag(_, Rating, Traffic, _, 'RISK_LOW_RATING_PEAK_TRAFFIC') :-
    Rating < 3.8,
    (Traffic == 'Jam' ; Traffic == 'High'), !.

% Rule 1C: High risk if multiple deliveries (> 1.0) during severe weather
dispatch_flag(_, _, _, Weather, 'RISK_MULTIPLE_DELIVERIES_BAD_WEATHER') :-
    (Weather == 'Stormy' ; Weather == 'Sandstorms' ; Weather == 'Fog'), !.

% Default: Order approved for standard dispatch
dispatch_flag(_, _, _, _, 'APPROVED').

% 2. WEATHER PENALTY RULES (MINUTES)
weather_penalty('Stormy', 15.0) :- !.
weather_penalty('Sandstorms', 12.0) :- !.
weather_penalty('Fog', 8.0) :- !.
weather_penalty('Windy', 5.0) :- !.
weather_penalty('Cloudy', 3.0) :- !.
weather_penalty(_, 0.0).

% 3. TRAFFIC PENALTY RULES (MINUTES)
traffic_penalty('Jam', 12.0) :- !.
traffic_penalty('High', 7.0) :- !.
traffic_penalty('Medium', 3.0) :- !.
traffic_penalty(_, 0.0).

% 4. FESTIVAL SURGE PENALTY RULES (MINUTES)
festival_penalty('Yes', 15.0) :- !.
festival_penalty(_, 0.0).

% 5. MULTIPLE DELIVERIES LOAD PENALTY RULES (MINUTES)
multi_delivery_penalty(NumDeliveries, Penalty) :-
    NumDeliveries > 0,
    Penalty is NumDeliveries * 4.0, !.
multi_delivery_penalty(_, 0.0).

% 6. MAIN HYBRID EVALUATION INFERENCE PREDICATE
evaluate_order(MlEta, Distance, Rating, Weather, Traffic, Festival, MultiDeliveries,
               [FinalHybridEta, TotalPrologPenalty, DispatchFlag]) :-

    % 1. Determine Dispatch Guardrail Flag
    dispatch_flag(Distance, Rating, Traffic, Weather, DispatchFlag),

    % 2. Calculate Individual Symbolic Adjustments
    weather_penalty(Weather, WPen),
    traffic_penalty(Traffic, TPen),
    festival_penalty(Festival, FPen),
    multi_delivery_penalty(MultiDeliveries, MPen),

    % 3. Aggregate Total Symbolic Penalty
    TotalPrologPenalty is WPen + TPen + FPen + MPen,

    % 4. Compute Final Hybrid ETA
    FinalHybridEta is MlEta + TotalPrologPenalty.

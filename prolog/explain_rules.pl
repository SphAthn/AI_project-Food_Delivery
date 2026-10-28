% PROLOG explanation layer for one order at a time
% File: prolog/explain_rules.pl
% Description: Takes one order number to get the informations about it ans its ML prediction. the engine applies the 
% same rules as the batch engine (dispatch_rules.pl) and prints a short, uderstandable explanation of the final ETA and of the dispatch decision.

:- module(explain_rules, [explain_order/7]).
% import the dispatch rules module
:- use_module(dispatch_rules).


% main predictate to explain a single order

explain_order(MlEta, Distance, Rating, Weather, Traffic, Festival, MultiDeliveries) :-
    % Same inference as the batch engine
    evaluate_order(MlEta, Distance, Rating, Weather, Traffic, Festival, MultiDeliveries, [FinalEta, TotalPenalty, Flag]),

    % Individual penalties
    weather_penalty(Weather, WPen),
    traffic_penalty(Traffic, TPen),
    festival_penalty(Festival, FPen),
    multi_delivery_penalty(MultiDeliveries, MPen),
    safety_buffer(Buffer),

    %  factors that changed the ETA:
    findall(Line,
            ( member(Factor-Pen, [ safety_buffer-Buffer,
                                   weather-WPen,
                                   traffic-TPen,
                                   festival-FPen,
                                   multiple_deliveries-MPen ]),
              Pen =\= 0,
              reason_text(Factor, MultiDeliveries, Text),
              signed(Pen, SPen),
              format(atom(Line), '~w min ~w', [SPen, Text])
            ),
            Lines),

    signed(TotalPenalty, STotal),

    format("~nOrder: ~1f km, courier rating ~1f, weather ~w, traffic ~w, festival ~w.~n",
           [Distance, Rating, Weather, Traffic, Festival]),

    ( Lines == []
    ->  format("The ML model predicts ~1f min and no adjustment is needed.~n", [MlEta])
    ;   format("The ML model predicts ~1f min and the adjustments add ~w min, so the ETA is ~1f min.~n",
               [MlEta, STotal, FinalEta]),
        forall(member(L, Lines), format("  ~w~n", [L]))
    ),

    flag_text(Flag, Decision, Tradeoff),
    format("~nDecision: ~w~n", [Decision]),
    ( Tradeoff == '' -> true ; format("~w~n", [Tradeoff]) ).


% helpers
signed(X, S) :- X > 0, !, format(atom(S), '+~1f', [X]).
signed(X, S) :- format(atom(S), '~1f', [X]).


% human readable explanations

reason_text(safety_buffer, _,'as a standard safety margin: the ML estimate alone is late for about one order out of two.').
reason_text(weather, _,'for the weather, which slows couriers down.').
reason_text(traffic, _, 'for the traffic, which slows couriers down.').
reason_text(festival, _, 'for the festival period, when the roads are busier.').
reason_text(multiple_deliveries, Multi, Text) :-( Multi =:= 1 -> Noun = order ; Noun = orders ), format(atom(Text), 'because the courier carries ~0f extra ~w at the same time.', [Multi, Noun]).

flag_text('APPROVED', 'Approved for standard dispatch, no safety rule triggered.','').
flag_text('REJECT_DISTANCE_EXCEEDED', 'Reject, the distance is above the 25 km safety limit.', 'This protects the courier from a very long ride and avoids a likely late delivery, but we lose the order.').
flag_text('RISK_LOW_RATING_PEAK_TRAFFIC', 'High risk, the courier has a low rating (below 3.8) and traffic is heavy.', 'If a better-rated courier is free, reassign the order, even if that delays another delivery. Otherwise keep the longer ETA so the courier does not feel pushed to rush.').
flag_text('RISK_MULTIPLE_DELIVERIES_BAD_WEATHER', 'High risk, several deliveries at once in severe weather.', 'Splitting them is safer for the courier and better for the food, but it needs more couriers and costs the company more.').
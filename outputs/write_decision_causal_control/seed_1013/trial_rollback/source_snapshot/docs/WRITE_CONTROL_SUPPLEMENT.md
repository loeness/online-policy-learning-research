
After selection, an additional declared causal control reruns `trial_rollback`
on the same three formal seeds/bout budgets. It retains trial A+C learning as
local A while forbidding persistent C writes. This separates benefits of the
trial gradient parameterization from benefits of accepted shared writes. It
does not change the selected rules or the original five-arm formal cohort;
its output is a separate `post_selection_causal_control` cohort and is not
eligible for selection.

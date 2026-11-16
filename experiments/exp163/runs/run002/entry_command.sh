set +e
sh /data/robot/p001/e163/o002/launch_command.sh
run002_outer_rc=$?
printf 'RUN002_ENTRY_RETURN_CODE=%s\n' "$run002_outer_rc"

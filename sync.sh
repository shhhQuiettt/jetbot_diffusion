USERNAME="jeanluc"
REMOTE_IP="192.168.1.44"
PATH_TO_PROJECT="/home/jeanluc/ks/jetbot_diffusion"
# Sync local folder to the remote server folder (where docker-compose.yml lives)
#
# watchman-make -p '**/*.py' --run 'rsync -avz ./ user@remote-ip:/path/to/project'
# find . -name "*.py" | entr rsync -avz ./ $USERNAME@$REMOTE_IP:$PATH_TO_PROJECT
while true; do
  find . -name "*" | entr -d rsync --exclude='.git/' --exclude='.venv/' -avz ./ $USERNAME@$REMOTE_IP:$PATH_TO_PROJECT
  sleep 1
done

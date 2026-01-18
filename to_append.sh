# Custom Aliases & Functions
tnew() {
    tmux new -s "$1"
}

tat() {
    tmux a -t "$1"
}

tkl() {
    tmux kill-session -t "$1"
}

alias gd='cd /root/myCodeLab/host/verl'

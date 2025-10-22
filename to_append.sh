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

export http_proxy="http://jdtcom:709a64b73eb3@10.119.176.202:3128"
export https_proxy="http://jdtcom:709a64b73eb3@10.119.176.202:3128"
export HTTP_PROXY="http://jdtcom:709a64b73eb3@10.119.176.202:3128"
export HTTPS_PROXY="http://jdtcom:709a64b73eb3@10.119.176.202:3128"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

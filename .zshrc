export PATH="/opt/homebrew/bin:$PATH"
eval "$(/opt/homebrew/bin/brew shellenv)"
eval "$(/opt/homebrew/bin/brew shellenv)"
export PATH="$PATH:/Users/amarnathmahato/mongodb-macos-aarch64-8.0.11/bin"
alias mongod='mongod --dbpath ~/mongodb/data/db'

export GOPATH=$HOME/go
export PATH=$PATH:/usr/local/go/bin:$GOPATH/bin

. "$HOME/.local/bin/env"

# Added by Antigravity
export PATH="/Users/amarnathmahato/.antigravity/antigravity/bin:$PATH"

# Added by Antigravity
export PATH="/Users/amarnathmahato/.antigravity/antigravity/bin:$PATH"
export PATH="/opt/homebrew/opt/expat/bin:$PATH"
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

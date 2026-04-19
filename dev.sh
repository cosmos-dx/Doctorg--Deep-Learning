#!/bin/bash
# DoctorG Development Script
# Usage: ./dev.sh start | stop | restart | status

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
BACKEND_PID_FILE="$PROJECT_DIR/.backend.pid"
FRONTEND_PID_FILE="$PROJECT_DIR/.frontend.pid"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

start_backend() {
  if [ -f "$BACKEND_PID_FILE" ] && kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
    echo -e "${YELLOW}Backend already running (PID $(cat "$BACKEND_PID_FILE"))${NC}"
    return
  fi

  echo -e "${GREEN}Starting backend...${NC}"
  cd "$BACKEND_DIR" || exit 1
  if [ ! -d "venv" ]; then
    python3 -m venv venv
  fi
  source venv/bin/activate
  echo -e "${YELLOW}Installing backend dependencies...${NC}"
  pip install -q -r requirements.txt
  
  PYTHONPATH="$BACKEND_DIR" uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --env-file "$PROJECT_DIR/.env" > "$PROJECT_DIR/logs/backend.log" 2>&1 &
  echo $! > "$BACKEND_PID_FILE"
  echo -e "${GREEN}Backend started (PID $!) → http://localhost:8000${NC}"
}

start_frontend() {
  if [ -f "$FRONTEND_PID_FILE" ] && kill -0 "$(cat "$FRONTEND_PID_FILE")" 2>/dev/null; then
    echo -e "${YELLOW}Frontend already running (PID $(cat "$FRONTEND_PID_FILE"))${NC}"
    return
  fi

  echo -e "${GREEN}Starting frontend...${NC}"
  cd "$FRONTEND_DIR" || exit 1
  
  if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
  fi

  npx next dev -p 3000 > "$PROJECT_DIR/logs/frontend.log" 2>&1 &
  echo $! > "$FRONTEND_PID_FILE"
  echo -e "${GREEN}Frontend started (PID $!) → http://localhost:3000${NC}"
}

stop_service() {
  local name=$1
  local pid_file=$2
  local port=$3

  if [ -f "$pid_file" ]; then
    local pid
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null
      sleep 1
      kill -9 "$pid" 2>/dev/null
    fi
    rm -f "$pid_file"
  fi

  lsof -ti :"$port" | xargs kill -9 2>/dev/null
  echo -e "${RED}$name stopped${NC}"
}

status() {
  echo ""
  echo "=== DoctorG Status ==="

  if [ -f "$BACKEND_PID_FILE" ] && kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
    echo -e "Backend:  ${GREEN}running${NC} (PID $(cat "$BACKEND_PID_FILE")) → http://localhost:8000"
  else
    echo -e "Backend:  ${RED}stopped${NC}"
  fi

  if [ -f "$FRONTEND_PID_FILE" ] && kill -0 "$(cat "$FRONTEND_PID_FILE")" 2>/dev/null; then
    echo -e "Frontend: ${GREEN}running${NC} (PID $(cat "$FRONTEND_PID_FILE")) → http://localhost:3000"
  else
    echo -e "Frontend: ${RED}stopped${NC}"
  fi

  echo ""
}

case "${1:-start}" in
  start)
    mkdir -p "$PROJECT_DIR/logs"
    start_backend
    sleep 2
    start_frontend
    sleep 2
    status
    echo "Logs: tail -f logs/backend.log  |  tail -f logs/frontend.log"
    ;;
  stop)
    stop_service "Backend" "$BACKEND_PID_FILE" 8000
    stop_service "Frontend" "$FRONTEND_PID_FILE" 3000
    status
    ;;
  restart)
    stop_service "Backend" "$BACKEND_PID_FILE" 8000
    stop_service "Frontend" "$FRONTEND_PID_FILE" 3000
    sleep 1
    mkdir -p "$PROJECT_DIR/logs"
    start_backend
    sleep 2
    start_frontend
    sleep 2
    status
    ;;
  status)
    status
    ;;
  logs)
    tail -f "$PROJECT_DIR/logs/backend.log" "$PROJECT_DIR/logs/frontend.log"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}"
    exit 1
    ;;
esac

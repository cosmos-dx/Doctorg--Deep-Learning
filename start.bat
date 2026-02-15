@echo off
echo 🚀 Starting DoctorG Medical AI...
echo.

REM Check if .env exists
if not exist .env (
    echo ❌ .env file not found!
    echo 📝 Creating from .env.example...
    copy .env.example .env
    echo ⚠️  Please edit .env file with your API keys before continuing
    echo    Required: OPENAI_API_KEY, POSTGRES_PASSWORD, JWT_SECRET
    exit /b 1
)

echo ✅ Environment file found
echo.

REM Build and start services
echo 🏗️  Building Docker containers...
docker-compose build

echo.
echo 🚀 Starting services...
docker-compose up -d

echo.
echo ⏳ Waiting for services to be ready...
timeout /t 10 /nobreak > nul

echo.
echo 🔍 Checking service health...

REM Check backend
curl -s http://localhost:8000/health > nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend is running at http://localhost:8000
) else (
    echo ⚠️  Backend may still be starting...
)

REM Check frontend
curl -s http://localhost:3000 > nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Frontend is running at http://localhost:3000
) else (
    echo ⚠️  Frontend may still be starting...
)

echo.
echo 📊 Service Status:
docker-compose ps

echo.
echo 🎉 DoctorG Medical AI is starting!
echo.
echo 📍 Access Points:
echo    Frontend:  http://localhost:3000
echo    Backend:   http://localhost:8000
echo    API Docs:  http://localhost:8000/docs
echo.
echo 📝 View logs:
echo    docker-compose logs -f
echo.
echo 🛑 Stop services:
echo    docker-compose down
echo.
pause

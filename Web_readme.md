# run backend : 
sbatch backend/run_backend.sh

# run frontend : 
cd frontend && npm run dev -- -p 3000 -H 0.0.0.0
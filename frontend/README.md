# OncoMorph 4D Frontend

npm install
npm run dev

Backend:
http://localhost:8000

The frontend uploads one NIfTI MRI per timepoint.

Backend API:
POST /api/upload-mri?timepoint=T1
GET /api/dimensions/{scan_id}
GET /api/slice/{scan_id}/{plane}/{index}?show_mask=true
GET /api/model/brain/{scan_id}
GET /api/model/tumor/{scan_id}
GET /api/mask/{scan_id}

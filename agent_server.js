const express = require('express');
const app = express();

// Middleware untuk mem-parsing JSON body
app.use(express.json());

const port = process.env.PORT || 3000;

// Endpoint untuk menerima sinyal trading
app.post('/submit_signal', (req, res) => {
  const payload = req.body;

  if (!payload || Object.keys(payload).length === 0) {
    console.error('❌ Menerima payload kosong atau tidak valid.');
    return res.status(400).json({ status: 'ERROR', message: 'Invalid or empty JSON payload.' });
  }

  console.log('✅ Sinyal diterima:', new Date().toISOString());
  console.log('   - Profil Strategi:', payload.profile_name || 'N/A');
  console.log('   - Simbol:', payload.symbol || 'N/A');
  console.log('   - Tipe Order:', payload.order_type || 'N/A');
  console.log('   - Skor:', payload.score !== undefined ? payload.score.toFixed(2) : 'N/A');
  console.log('   - Payload Lengkap:', JSON.stringify(payload, null, 2));

  // Kirim respons berhasil
  res.status(200).json({ status: 'SUCCESS', message: 'Signal received successfully.' });
});

// Handle 404 untuk path lainnya
app.use((req, res) => {
  res.status(404).send('Endpoint not found. Use POST to /submit_signal.');
});

app.listen(port, () => {
  console.log(`🚀 Server "Smart Agent" (Express) berjalan di http://localhost:${port}`);
  console.log('Menunggu sinyal di endpoint /submit_signal...');
});

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::time::Instant;

use eframe::egui;
use serde::{Deserialize};

#[derive(Default)]
struct AppState {
    event: String,
    ifos: String,
    tpad: f32,
    flow: f32,
    fhigh: f32,
    gpu_device: i32,
    out_dir: String,
    running: bool,
    log: String,
    result_json: Option<PathBuf>,
    alpha: Option<f64>,
    k: Option<f64>,
    mc_solar: Option<f64>,
    last_run_ms: Option<u128>,
}

#[derive(Deserialize, Debug)]
struct FitResults {
    event: String,
    fs: f64,
    flow: f64,
    fhigh: f64,
    alpha: f64,
    K: f64,
    Mc_solar: f64,
}

impl Default for FitResults { fn default() -> Self { Self { event: String::new(), fs:0.0, flow:0.0, fhigh:0.0, alpha:0.0, K:0.0, Mc_solar:0.0 } } }

impl AppState {
    fn run_discover(&mut self) {
        if self.running { return; }
        self.running = true;
        self.log.clear();
        let event = if self.event.is_empty() { "GW150914".to_string() } else { self.event.clone() };
        let out = if self.out_dir.is_empty() { format!("outputs/{}", event.to_lowercase()) } else { self.out_dir.clone() };
        let ifos: Vec<&str> = self.ifos.split_whitespace().collect();
        let mut args = vec![
            "discover-inspiral".to_string(),
            "--event".to_string(), event.clone(),
            "--tpad".to_string(), self.tpad.to_string(),
            "--flow".to_string(), self.flow.to_string(),
            "--fhigh".to_string(), self.fhigh.to_string(),
            "--out".to_string(), out.clone(),
            "--gpu-device".to_string(), self.gpu_device.to_string(),
            "--save-plots".to_string(), "true".to_string(),
            "--save-csv".to_string(), "true".to_string(),
        ];
        for ifo in ifos { args.push("--ifo".to_string()); args.push(ifo.to_string()); }

        let start = Instant::now();
        let mut child = Command::new("derivegr")
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to spawn derivegr");
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let mut reader_out = BufReader::new(stdout);
        let mut reader_err = BufReader::new(stderr);

        // Read both streams non-blocking-ish in this simple example
        std::thread::spawn({
            let log_ptr = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
            move || { let _ = log_ptr; }
        });

        let mut buf = String::new();
        while let Ok(n) = reader_out.read_line(&mut buf) {
            if n == 0 { break; }
            self.log.push_str(&buf);
            buf.clear();
        }
        while let Ok(n) = reader_err.read_line(&mut buf) {
            if n == 0 { break; }
            self.log.push_str(&buf);
            buf.clear();
        }
        let _ = child.wait();
        self.last_run_ms = Some(start.elapsed().as_millis());

        // Try to load results
        let result_path = Path::new(&out).join("fit_results.json");
        if result_path.exists() {
            if let Ok(txt) = std::fs::read_to_string(&result_path) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                    self.alpha = v.get("alpha").and_then(|x| x.as_f64());
                    self.k = v.get("K").and_then(|x| x.as_f64());
                    self.mc_solar = v.get("Mc_solar").and_then(|x| x.as_f64());
                    self.result_json = Some(result_path.to_path_buf());
                }
            }
        }
        self.running = false;
    }
}

impl eframe::App for AppState {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("DeriveGR – GPU Inspiral Discovery");
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Event:");
                ui.text_edit_singleline(&mut self.event);
                if self.event.is_empty() { ui.label("(e.g., GW150914)"); }
            });
            ui.horizontal(|ui| {
                ui.label("IFOs (space-separated):");
                ui.text_edit_singleline(&mut self.ifos);
                if self.ifos.is_empty() { ui.label("H1 L1"); }
            });
            ui.horizontal(|ui| {
                ui.label("tpad(s)"); ui.add(egui::DragValue::new(&mut self.tpad).speed(0.5));
                ui.label("flow(Hz)"); ui.add(egui::DragValue::new(&mut self.flow).speed(1.0));
                ui.label("fhigh(Hz)"); ui.add(egui::DragValue::new(&mut self.fhigh).speed(1.0));
            });
            ui.horizontal(|ui| {
                ui.label("GPU device"); ui.add(egui::DragValue::new(&mut self.gpu_device).clamp_range(0..=8));
                ui.label("Out dir"); ui.text_edit_singleline(&mut self.out_dir);
            });

            if ui.add_enabled(!self.running, egui::Button::new("Run discovery")).clicked() {
                let mut defaults = self.clone();
                if defaults.event.is_empty() { defaults.event = "GW150914".into(); }
                if defaults.ifos.is_empty() { defaults.ifos = "H1 L1".into(); }
                if defaults.tpad == 0.0 { defaults.tpad = 16.0; }
                if defaults.flow == 0.0 { defaults.flow = 30.0; }
                if defaults.fhigh == 0.0 { defaults.fhigh = 350.0; }
                *self = defaults;
                self.run_discover();
            }

            if let Some(ms) = self.last_run_ms { ui.label(format!("Last run: {} ms", ms)); }
            if let Some(a) = self.alpha { ui.label(format!("alpha: {:.4}", a)); }
            if let Some(k) = self.k { ui.label(format!("K: {:.6e}", k)); }
            if let Some(mc) = self.mc_solar { ui.label(format!("M_chirp: {:.2} M_sun", mc)); }

            ui.separator();
            ui.label("Log:");
            egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                ui.monospace(&self.log);
            });

            if let Some(j) = &self.result_json {
                ui.horizontal(|ui| {
                    if ui.button("Open results folder").clicked() {
                        let _ = open::that(j.parent().unwrap());
                    }
                    if ui.button("Open loglog plot").clicked() {
                        let p = j.parent().unwrap().join("loglog_fit.png");
                        let _ = open::that(p);
                    }
                });
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let mut app = AppState::default();
    app.event = "GW150914".into();
    app.ifos = "H1 L1".into();
    app.tpad = 16.0;
    app.flow = 30.0;
    app.fhigh = 350.0;
    app.gpu_device = 0;
    app.out_dir = String::new();

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "DeriveGR GUI",
        options,
        Box::new(|_cc| Box::new(app)),
    )
}


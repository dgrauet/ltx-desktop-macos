import SwiftUI

struct TrainingRunView: View {
    @EnvironmentObject var vm: TrainingViewModel
    @EnvironmentObject var backendService: BackendService

    @State private var runToDelete: TrainingRun?
    @State private var showDeleteConfirmation = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // MARK: - Live Progress

            GroupBox("Training Progress") {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text(vm.liveStatus.isEmpty ? "Idle" : vm.liveStatus)
                            .font(.subheadline)
                            .foregroundStyle(vm.isTraining ? .primary : .secondary)
                        Spacer()
                        if vm.livePeakGb > 0 {
                            Text("Peak: \(vm.livePeakGb, specifier: "%.1f") GB")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                        }
                    }

                    if vm.liveTotal > 0 {
                        let progress = Double(vm.liveStep) / Double(max(vm.liveTotal, 1))
                        ProgressView(value: progress)
                            .tint(.accentColor)
                        HStack {
                            Text("Step \(vm.liveStep) / \(vm.liveTotal)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                            Spacer()
                            Text("\(Int(progress * 100))%")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                        }
                    } else {
                        ProgressView(value: 0.0)
                            .tint(.accentColor)
                            .opacity(vm.isTraining ? 1 : 0.3)
                    }

                    if vm.isTraining {
                        Button(role: .destructive) {
                            Task { await vm.cancelTraining(using: backendService) }
                        } label: {
                            Label("Cancel Training", systemImage: "xmark.circle.fill")
                        }
                        .buttonStyle(.bordered)
                        .tint(.red)
                    }
                }
                .padding(4)
            }

            // MARK: - Error

            if let error = vm.errorMessage {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                .padding(8)
                .background(Color.red.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }

            // MARK: - Past Runs

            GroupBox("Past Runs") {
                if vm.runs.isEmpty {
                    Text("No training runs yet.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 8)
                } else {
                    VStack(spacing: 0) {
                        ForEach(vm.runs) { run in
                            RunRowView(run: run) {
                                runToDelete = run
                                showDeleteConfirmation = true
                            }
                            Divider()
                        }
                    }
                }
            }

            // MARK: - Note

            HStack(spacing: 6) {
                Image(systemName: "info.circle")
                    .foregroundStyle(.secondary)
                    .font(.caption)
                Text("Produced LoRAs are automatically available in the Generation LoRA picker.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 4)
        }
        .padding()
        .task {
            await vm.loadRuns(using: backendService)
        }
        .confirmationDialog(
            "Delete Run",
            isPresented: $showDeleteConfirmation,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                if let run = runToDelete {
                    Task { await vm.deleteRun(run, using: backendService) }
                }
                runToDelete = nil
            }
            Button("Cancel", role: .cancel) {
                runToDelete = nil
            }
        } message: {
            if let run = runToDelete {
                Text("Delete run \(run.runId.prefix(8))? This cannot be undone.")
            }
        }
    }
}

// MARK: - Run Row

private struct RunRowView: View {
    let run: TrainingRun
    let onDelete: () -> Void

    var statusColor: Color {
        switch run.status {
        case "done": return .green
        case "error", "failed": return .red
        case "cancelled": return .orange
        default: return .secondary
        }
    }

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: statusIcon)
                .foregroundStyle(statusColor)
                .frame(width: 16)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(run.runId.prefix(8))
                        .font(.caption)
                        .fontDesign(.monospaced)
                    Text("·")
                        .foregroundStyle(.secondary)
                    Text(run.status.capitalized)
                        .font(.caption)
                        .foregroundStyle(statusColor)
                }
                HStack(spacing: 6) {
                    Text("Dataset: \(run.datasetId)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if let peakGb = run.peakMemGb {
                        Text("·")
                            .foregroundStyle(.secondary)
                        Text("Peak: \(peakGb, specifier: "%.1f") GB")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                }
            }

            Spacer()

            if run.loraPath != nil {
                Image(systemName: "checkmark.seal.fill")
                    .foregroundStyle(.green)
                    .font(.caption)
                    .help("LoRA available in Generation picker")
            }

            Button(role: .destructive) {
                onDelete()
            } label: {
                Image(systemName: "trash")
                    .font(.caption)
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.red.opacity(0.7))
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 4)
    }

    private var statusIcon: String {
        switch run.status {
        case "done": return "checkmark.circle.fill"
        case "error", "failed": return "exclamationmark.circle.fill"
        case "cancelled": return "minus.circle.fill"
        case "running": return "arrow.trianglehead.2.clockwise.rotate.90.circle"
        default: return "circle"
        }
    }
}

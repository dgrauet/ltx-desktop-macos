import SwiftUI

// MARK: - TrainingConfigView

struct TrainingConfigView: View {
    @EnvironmentObject var backendService: BackendService
    @EnvironmentObject var vm: TrainingViewModel

    @State private var isRunningPreflight = false
    @State private var isStartingTraining = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            headerBar
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    modeSection
                    parametersSection
                    preflightSection
                    startSection
                    stabilityNote
                }
                .padding(16)
            }
        }
    }

    // MARK: - Header

    private var headerBar: some View {
        HStack {
            Text("Training Config")
                .font(.title2)
                .fontWeight(.semibold)
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.top, 12)
        .padding(.bottom, 10)
    }

    // MARK: - Mode Section

    private var modeSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Training Mode")
            Picker("Mode", selection: Binding(
                get: { vm.lowRam ? 1 : 0 },
                set: { newVal in
                    vm.lowRam = (newVal == 1)
                    vm.applyDefaults(lowRam: vm.lowRam)
                }
            )) {
                Text("Normal").tag(0)
                Text("Low-RAM (32 GB)").tag(1)
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            Text(vm.lowRam
                 ? "Optimized for 32 GB. Rank and step defaults are reduced to fit within memory."
                 : "Standard training. Recommended for 64 GB or higher.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Parameters Section

    private var parametersSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionLabel("Parameters")

            // Rank
            HStack {
                Text("Rank")
                    .frame(width: 80, alignment: .leading)
                Stepper(value: $vm.rank, in: 4...256, step: 4) {
                    Text("\(vm.rank)")
                        .monospacedDigit()
                        .frame(width: 40, alignment: .trailing)
                }
            }

            // Steps
            HStack {
                Text("Steps")
                    .frame(width: 80, alignment: .leading)
                Stepper(value: $vm.steps, in: 50...5000, step: 50) {
                    Text("\(vm.steps)")
                        .monospacedDigit()
                        .frame(width: 50, alignment: .trailing)
                }
            }
        }
    }

    // MARK: - Preflight Section

    private var preflightSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionLabel("Preflight Check")

            Button {
                isRunningPreflight = true
                Task {
                    await vm.runPreflight(using: backendService)
                    isRunningPreflight = false
                }
            } label: {
                if isRunningPreflight {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("Checking…")
                    }
                } else {
                    Label("Run preflight", systemImage: "checklist")
                }
            }
            .buttonStyle(.bordered)
            .disabled(vm.selectedDatasetId == nil || isRunningPreflight || vm.isTraining)

            if let result = vm.preflight {
                preflightBanner(result)
            }
        }
    }

    private func preflightBanner(_ result: PreflightResult) -> some View {
        let (color, icon, label): (Color, String, String) = {
            switch result.verdict {
            case "ok":
                return (.green, "checkmark.circle.fill", "ok")
            case "risky":
                return (.orange, "exclamationmark.triangle.fill", "risky")
            default:
                return (.red, "xmark.octagon.fill", "oom")
            }
        }()

        return HStack(spacing: 10) {
            Image(systemName: icon)
                .foregroundStyle(color)
            VStack(alignment: .leading, spacing: 2) {
                Text("Peak \u{2248} \(result.peakGb, specifier: "%.1f") GB — \(label)")
                    .font(.callout)
                    .fontWeight(.medium)
                    .foregroundStyle(color)
                if result.verdict == "oom" {
                    Text("Estimated peak exceeds available memory. Reduce rank or steps, or enable Low-RAM mode.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if result.verdict == "risky" {
                    Text("Memory will be tight. Close other apps and consider reducing rank or steps.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(color.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(color.opacity(0.25), lineWidth: 1)
        )
    }

    // MARK: - Start Section

    private var startSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionLabel("Start Training")

            if let error = vm.errorMessage {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(error)
                        .font(.callout)
                        .foregroundStyle(.red)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(Color.red.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 7))
                .overlay(
                    RoundedRectangle(cornerRadius: 7)
                        .strokeBorder(Color.red.opacity(0.25), lineWidth: 1)
                )
            }

            HStack(spacing: 12) {
                Button {
                    isStartingTraining = true
                    Task {
                        await vm.startTraining(using: backendService)
                        isStartingTraining = false
                    }
                } label: {
                    if vm.isTraining || isStartingTraining {
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.small)
                            Text("Training…")
                        }
                    } else {
                        Label("Start training", systemImage: "play.fill")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(vm.selectedDatasetId == nil || vm.isTraining || isStartingTraining)

                if vm.isTraining {
                    Button {
                        Task { await vm.cancelTraining(using: backendService) }
                    } label: {
                        Label("Cancel", systemImage: "stop.fill")
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                }
            }

            if vm.isTraining {
                trainingProgressBar
            }

            if vm.selectedDatasetId == nil {
                Text("Select a dataset in the Dataset Builder tab before starting.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var trainingProgressBar: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(vm.liveStatus)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
                if vm.livePeakGb > 0 {
                    Text("Peak \u{2248} \(vm.livePeakGb, specifier: "%.1f") GB")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
            }

            let progress = vm.liveTotal > 0
                ? Double(vm.liveStep) / Double(vm.liveTotal)
                : 0.0

            ProgressView(value: progress)
                .progressViewStyle(.linear)
                .tint(.accentColor)

            Text("Step \(vm.liveStep) / \(vm.liveTotal)")
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
        .padding(.top, 4)
    }

    // MARK: - Stability Note

    private var stabilityNote: some View {
        HStack(spacing: 8) {
            Image(systemName: "info.circle")
                .foregroundStyle(.secondary)
            Text("Close other GPU-heavy apps for best stability.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.top, 4)
    }

    // MARK: - Helpers

    private func sectionLabel(_ text: String) -> some View {
        Text(text)
            .font(.headline)
            .fontWeight(.semibold)
    }
}

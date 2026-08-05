// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaor contributors

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace KaorPortableAssembler
{
    [DataContract]
    internal sealed class ReleasePart
    {
        [DataMember(Name = "name")]
        public string Name { get; set; }

        [DataMember(Name = "size_bytes")]
        public long SizeBytes { get; set; }

        [DataMember(Name = "sha256")]
        public string Sha256 { get; set; }
    }

    [DataContract]
    internal sealed class ReleaseFile
    {
        [DataMember(Name = "name")]
        public string Name { get; set; }

        [DataMember(Name = "size_bytes")]
        public long SizeBytes { get; set; }

        [DataMember(Name = "sha256")]
        public string Sha256 { get; set; }
    }

    [DataContract]
    internal sealed class ReleaseManifest
    {
        [DataMember(Name = "schema_version")]
        public int SchemaVersion { get; set; }

        [DataMember(Name = "product")]
        public string Product { get; set; }

        [DataMember(Name = "version")]
        public string Version { get; set; }

        [DataMember(Name = "runtime_profile")]
        public string RuntimeProfile { get; set; }

        [DataMember(Name = "archive_name")]
        public string ArchiveName { get; set; }

        [DataMember(Name = "archive_size_bytes")]
        public long ArchiveSizeBytes { get; set; }

        [DataMember(Name = "archive_sha256")]
        public string ArchiveSha256 { get; set; }

        [DataMember(Name = "unpacked_size_bytes")]
        public long UnpackedSizeBytes { get; set; }

        [DataMember(Name = "package_directory")]
        public string PackageDirectory { get; set; }

        [DataMember(Name = "part_size_bytes")]
        public long PartSizeBytes { get; set; }

        [DataMember(Name = "parts")]
        public List<ReleasePart> Parts { get; set; }

        [DataMember(Name = "assembler")]
        public ReleaseFile Assembler { get; set; }

        [DataMember(Name = "assembler_source")]
        public ReleaseFile AssemblerSource { get; set; }
    }

    internal sealed class InstallResult
    {
        public string TargetDirectory { get; set; }
    }

    internal static class ReleaseInstaller
    {
        private const int BufferSize = 8 * 1024 * 1024;

        public static ReleaseManifest ReadManifest(string manifestPath)
        {
            var manifestFile = new FileInfo(manifestPath);
            if (!manifestFile.Exists)
            {
                throw new FileNotFoundException("Release manifest was not found.", manifestPath);
            }
            if (manifestFile.Length <= 0 || manifestFile.Length > 1024 * 1024)
            {
                throw new InvalidDataException("The release manifest has an invalid size.");
            }

            var serializer = new DataContractJsonSerializer(typeof(ReleaseManifest));
            var json = File.ReadAllText(manifestPath, Encoding.UTF8);
            var jsonBytes = Encoding.UTF8.GetBytes(json);
            using (var input = new MemoryStream(jsonBytes, false))
            {
                var manifest = serializer.ReadObject(input) as ReleaseManifest;
                ValidateManifest(manifest);
                return manifest;
            }
        }

        public static void VerifyOnly(
            string manifestPath,
            Action<int, string> report,
            Func<bool> cancelled)
        {
            var manifest = ReadManifest(manifestPath);
            var baseDirectory = Path.GetDirectoryName(Path.GetFullPath(manifestPath));
            var requiredSpace = checked(manifest.ArchiveSizeBytes + (256L * 1024L * 1024L));
            EnsureFreeSpace(baseDirectory, requiredSpace);
            var temporaryArchive = Path.Combine(
                baseDirectory,
                "." + manifest.ArchiveName + ".verify-" + Guid.NewGuid().ToString("N"));
            Exception operationError = null;
            try
            {
                Assemble(manifest, baseDirectory, temporaryArchive, report, cancelled);
                report(100, "All parts and the complete archive are valid.");
            }
            catch (Exception error)
            {
                operationError = error;
                throw;
            }
            finally
            {
                if (operationError == null)
                {
                    DeleteGeneratedFile(temporaryArchive);
                }
                else
                {
                    TryDeleteGeneratedFile(temporaryArchive);
                }
            }
        }

        public static InstallResult Install(
            string manifestPath,
            string destinationBase,
            Action<int, string> report,
            Func<bool> cancelled)
        {
            var manifest = ReadManifest(manifestPath);
            var partDirectory = Path.GetDirectoryName(Path.GetFullPath(manifestPath));
            destinationBase = Path.GetFullPath(destinationBase);
            Directory.CreateDirectory(destinationBase);

            var targetDirectory = Path.Combine(destinationBase, manifest.PackageDirectory);
            if (Directory.Exists(targetDirectory) || File.Exists(targetDirectory))
            {
                throw new IOException(
                    "The destination already exists: " + targetDirectory +
                    Environment.NewLine + "Rename the existing folder before installing this release.");
            }

            var requiredSpace = checked(
                manifest.ArchiveSizeBytes + manifest.UnpackedSizeBytes + (512L * 1024L * 1024L));
            EnsureFreeSpace(destinationBase, requiredSpace);

            var temporaryArchive = Path.Combine(
                destinationBase,
                "." + manifest.ArchiveName + ".assembling-" + Guid.NewGuid().ToString("N"));
            var stagingDirectory = Path.Combine(
                destinationBase,
                ".kaor-extract-" + Guid.NewGuid().ToString("N"));
            Exception operationError = null;
            try
            {
                Assemble(manifest, partDirectory, temporaryArchive, report, cancelled);
                ThrowIfCancelled(cancelled);
                ExtractSafely(manifest, temporaryArchive, stagingDirectory, report, cancelled);

                var stagedPackage = Path.Combine(stagingDirectory, manifest.PackageDirectory);
                if (!File.Exists(Path.Combine(stagedPackage, "Kaor.exe")))
                {
                    throw new InvalidDataException("The archive did not contain the expected Kaor.exe.");
                }
                MoveDirectoryWithRetry(stagedPackage, targetDirectory);
                report(100, "Kaor was extracted successfully.");
                return new InstallResult { TargetDirectory = targetDirectory };
            }
            catch (Exception error)
            {
                operationError = error;
                throw;
            }
            finally
            {
                if (operationError == null)
                {
                    Exception cleanupError = null;
                    try
                    {
                        DeleteGeneratedFile(temporaryArchive);
                    }
                    catch (Exception error)
                    {
                        cleanupError = error;
                    }
                    try
                    {
                        DeleteGeneratedDirectory(stagingDirectory);
                    }
                    catch (Exception error)
                    {
                        if (cleanupError == null)
                        {
                            cleanupError = error;
                        }
                    }
                    if (cleanupError != null)
                    {
                        throw cleanupError;
                    }
                }
                else
                {
                    TryDeleteGeneratedFile(temporaryArchive);
                    TryDeleteGeneratedDirectory(stagingDirectory);
                }
            }
        }

        private static void Assemble(
            ReleaseManifest manifest,
            string partDirectory,
            string outputPath,
            Action<int, string> report,
            Func<bool> cancelled)
        {
            long completedBytes = 0;
            var buffer = new byte[BufferSize];
            using (var archiveHash = SHA256.Create())
            using (var output = new FileStream(outputPath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                foreach (var part in manifest.Parts)
                {
                    ThrowIfCancelled(cancelled);
                    var partPath = ResolveSiblingFile(partDirectory, part.Name);
                    var info = new FileInfo(partPath);
                    if (!info.Exists)
                    {
                        throw new FileNotFoundException("Release part is missing.", partPath);
                    }
                    if (info.Length != part.SizeBytes)
                    {
                        throw new InvalidDataException("Release part size mismatch: " + part.Name);
                    }

                    report(
                        Percent(completedBytes, manifest.ArchiveSizeBytes, 0, 55),
                        "Verifying and assembling " + part.Name);
                    using (var partHash = SHA256.Create())
                    using (var input = new FileStream(partPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                    {
                        int read;
                        while ((read = input.Read(buffer, 0, buffer.Length)) > 0)
                        {
                            ThrowIfCancelled(cancelled);
                            partHash.TransformBlock(buffer, 0, read, buffer, 0);
                            archiveHash.TransformBlock(buffer, 0, read, buffer, 0);
                            output.Write(buffer, 0, read);
                            completedBytes += read;
                        }
                        partHash.TransformFinalBlock(new byte[0], 0, 0);
                        if (!FixedEquals(ToHex(partHash.Hash), part.Sha256))
                        {
                            throw new InvalidDataException("Release part SHA-256 mismatch: " + part.Name);
                        }
                    }
                }
                archiveHash.TransformFinalBlock(new byte[0], 0, 0);
                output.Flush(true);

                if (output.Length != manifest.ArchiveSizeBytes)
                {
                    throw new InvalidDataException("Assembled archive size mismatch.");
                }
                if (!FixedEquals(ToHex(archiveHash.Hash), manifest.ArchiveSha256))
                {
                    throw new InvalidDataException("Assembled archive SHA-256 mismatch.");
                }
            }
        }

        private static void ExtractSafely(
            ReleaseManifest manifest,
            string archivePath,
            string stagingDirectory,
            Action<int, string> report,
            Func<bool> cancelled)
        {
            Directory.CreateDirectory(stagingDirectory);
            var stagingPrefix = Path.GetFullPath(stagingDirectory)
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
                + Path.DirectorySeparatorChar;
            using (var archive = ZipFile.OpenRead(archivePath))
            {
                var count = archive.Entries.Count;
                long declaredBytes = 0;
                foreach (var entry in archive.Entries)
                {
                    declaredBytes = checked(declaredBytes + entry.Length);
                }
                if (declaredBytes != manifest.UnpackedSizeBytes)
                {
                    throw new InvalidDataException("The archive unpacked size does not match the release manifest.");
                }

                var buffer = new byte[BufferSize];
                var destinations = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                long extractedBytes = 0;
                for (var index = 0; index < count; index++)
                {
                    ThrowIfCancelled(cancelled);
                    var entry = archive.Entries[index];
                    if (((entry.ExternalAttributes >> 16) & 0xF000) == 0xA000)
                    {
                        throw new InvalidDataException("Symbolic links are not accepted in this release archive.");
                    }

                    ValidateArchiveEntryName(entry.FullName);
                    var relative = entry.FullName.Replace('/', Path.DirectorySeparatorChar);
                    var destination = Path.GetFullPath(Path.Combine(stagingDirectory, relative));
                    if (!destination.StartsWith(stagingPrefix, StringComparison.OrdinalIgnoreCase))
                    {
                        throw new InvalidDataException("Unsafe archive path: " + entry.FullName);
                    }
                    if (!destinations.Add(destination))
                    {
                        throw new InvalidDataException("Duplicate archive path: " + entry.FullName);
                    }

                    if (String.IsNullOrEmpty(entry.Name))
                    {
                        if (entry.Length != 0)
                        {
                            throw new InvalidDataException("Directory entry contains unexpected data: " + entry.FullName);
                        }
                        Directory.CreateDirectory(destination);
                    }
                    else
                    {
                        var parent = Path.GetDirectoryName(destination);
                        if (!String.IsNullOrEmpty(parent))
                        {
                            Directory.CreateDirectory(parent);
                        }
                        using (var input = entry.Open())
                        using (var output = new FileStream(
                            destination, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                        {
                            long entryBytes = 0;
                            int read;
                            while ((read = input.Read(buffer, 0, buffer.Length)) > 0)
                            {
                                ThrowIfCancelled(cancelled);
                                entryBytes = checked(entryBytes + read);
                                extractedBytes = checked(extractedBytes + read);
                                if (entryBytes > entry.Length || extractedBytes > manifest.UnpackedSizeBytes)
                                {
                                    throw new InvalidDataException("Archive entry exceeds its declared size: " + entry.FullName);
                                }
                                output.Write(buffer, 0, read);
                            }
                            if (entryBytes != entry.Length)
                            {
                                throw new InvalidDataException("Archive entry size mismatch: " + entry.FullName);
                            }
                        }
                    }
                    report(55 + Percent(index + 1, Math.Max(1, count), 0, 44), "Extracting Kaor files");
                }
                if (extractedBytes != manifest.UnpackedSizeBytes)
                {
                    throw new InvalidDataException("The extracted size does not match the release manifest.");
                }
            }
        }

        private static void ValidateManifest(ReleaseManifest manifest)
        {
            if (manifest == null || manifest.SchemaVersion != 1)
            {
                throw new InvalidDataException("Unsupported release manifest.");
            }
            if (manifest.Product != "Kaor")
            {
                throw new InvalidDataException("This is not a Kaor release manifest.");
            }
            if (manifest.RuntimeProfile != "nvidia-cu126")
            {
                throw new InvalidDataException("This assembler only accepts the NVIDIA CUDA release.");
            }
            ValidateFileName(manifest.ArchiveName, "archive_name");
            ValidateFileName(manifest.PackageDirectory, "package_directory");
            if (manifest.PackageDirectory != "Kaor-Windows-x64-NVIDIA" ||
                manifest.ArchiveName != manifest.PackageDirectory + ".zip")
            {
                throw new InvalidDataException("Unexpected NVIDIA release archive name.");
            }
            if (manifest.ArchiveSizeBytes <= 0 || manifest.UnpackedSizeBytes <= 0)
            {
                throw new InvalidDataException("Invalid archive size metadata.");
            }
            if (manifest.PartSizeBytes <= 0 || manifest.PartSizeBytes >= 2L * 1024L * 1024L * 1024L)
            {
                throw new InvalidDataException("Invalid release part size limit.");
            }
            ValidateHash(manifest.ArchiveSha256, "archive_sha256");
            if (manifest.Parts == null || manifest.Parts.Count < 2 || manifest.Parts.Count > 999)
            {
                throw new InvalidDataException("A split release must contain between 2 and 999 parts.");
            }

            ValidateReleaseFile(manifest.Assembler, "assembler");
            ValidateReleaseFile(manifest.AssemblerSource, "assembler_source");

            long total = 0;
            var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (var index = 0; index < manifest.Parts.Count; index++)
            {
                var part = manifest.Parts[index];
                if (part == null)
                {
                    throw new InvalidDataException("Invalid release part entry.");
                }
                ValidateFileName(part.Name, "part name");
                ValidateHash(part.Sha256, "part sha256");
                if (part.SizeBytes <= 0 || part.SizeBytes > manifest.PartSizeBytes)
                {
                    throw new InvalidDataException("Invalid release part size: " + part.Name);
                }
                if (!names.Add(part.Name))
                {
                    throw new InvalidDataException("Duplicate release part: " + part.Name);
                }
                var expectedName = manifest.ArchiveName + "." + (index + 1).ToString("D3");
                if (part.Name != expectedName)
                {
                    throw new InvalidDataException("Unexpected release part order or name: " + part.Name);
                }
                total = checked(total + part.SizeBytes);
            }
            if (total != manifest.ArchiveSizeBytes)
            {
                throw new InvalidDataException("Part sizes do not equal the complete archive size.");
            }
        }

        private static void ValidateReleaseFile(ReleaseFile file, string field)
        {
            if (file == null || file.SizeBytes <= 0)
            {
                throw new InvalidDataException("Invalid " + field + " metadata.");
            }
            ValidateFileName(file.Name, field + " name");
            ValidateHash(file.Sha256, field + " sha256");
        }

        private static string ResolveSiblingFile(string directory, string name)
        {
            ValidateFileName(name, "file name");
            return Path.Combine(directory, name);
        }

        private static void ValidateFileName(string value, string field)
        {
            if (String.IsNullOrWhiteSpace(value) || Path.GetFileName(value) != value ||
                value == "." || value == "..")
            {
                throw new InvalidDataException("Invalid " + field + ".");
            }
        }

        private static void ValidateHash(string value, string field)
        {
            if (String.IsNullOrEmpty(value) || value.Length != 64)
            {
                throw new InvalidDataException("Invalid " + field + ".");
            }
            for (var index = 0; index < value.Length; index++)
            {
                var current = value[index];
                if (!((current >= '0' && current <= '9') ||
                    (current >= 'a' && current <= 'f') ||
                    (current >= 'A' && current <= 'F')))
                {
                    throw new InvalidDataException("Invalid " + field + ".");
                }
            }
        }

        private static void ValidateArchiveEntryName(string value)
        {
            if (String.IsNullOrEmpty(value) || value.IndexOf(':') >= 0)
            {
                throw new InvalidDataException("Unsafe Windows archive name: " + value);
            }

            var segments = value.Replace('\\', '/').Split('/');
            for (var index = 0; index < segments.Length; index++)
            {
                var segment = segments[index];
                if (segment.Length == 0)
                {
                    if (index == segments.Length - 1)
                    {
                        continue;
                    }
                    throw new InvalidDataException("Unsafe Windows archive name: " + value);
                }
                if (segment == "." || segment == ".." ||
                    segment.EndsWith(".", StringComparison.Ordinal) ||
                    segment.EndsWith(" ", StringComparison.Ordinal))
                {
                    throw new InvalidDataException("Unsafe Windows archive name: " + value);
                }
                for (var characterIndex = 0; characterIndex < segment.Length; characterIndex++)
                {
                    var character = segment[characterIndex];
                    if (character < 32 || "<>\"|?*".IndexOf(character) >= 0)
                    {
                        throw new InvalidDataException("Unsafe Windows archive name: " + value);
                    }
                }

                var extensionIndex = segment.IndexOf('.');
                var baseName = (extensionIndex >= 0 ? segment.Substring(0, extensionIndex) : segment)
                    .ToUpperInvariant();
                if (baseName == "CON" || baseName == "PRN" || baseName == "AUX" || baseName == "NUL" ||
                    baseName == "CONIN$" || baseName == "CONOUT$" ||
                    (baseName.Length == 4 &&
                        (baseName.StartsWith("COM", StringComparison.Ordinal) ||
                         baseName.StartsWith("LPT", StringComparison.Ordinal)) &&
                        baseName[3] >= '1' && baseName[3] <= '9'))
                {
                    throw new InvalidDataException("Unsafe Windows archive name: " + value);
                }
            }
        }

        private static void EnsureFreeSpace(string path, long requiredBytes)
        {
            var root = Path.GetPathRoot(Path.GetFullPath(path));
            var drive = new DriveInfo(root);
            if (drive.AvailableFreeSpace < requiredBytes)
            {
                throw new IOException(
                    "Not enough free disk space. Required: " + FormatBytes(requiredBytes) +
                    "; available: " + FormatBytes(drive.AvailableFreeSpace) + ".");
            }
        }

        private static int Percent(long value, long total, int offset, int span)
        {
            if (total <= 0)
            {
                return offset;
            }
            return offset + (int)Math.Min(span, (value * span) / total);
        }

        private static void MoveDirectoryWithRetry(string source, string destination)
        {
            Exception lastError = null;
            const int attempts = 8;
            for (var attempt = 1; attempt <= attempts; attempt++)
            {
                try
                {
                    Directory.Move(source, destination);
                    return;
                }
                catch (IOException error)
                {
                    lastError = error;
                }
                catch (UnauthorizedAccessException error)
                {
                    lastError = error;
                }

                if (Directory.Exists(destination) || File.Exists(destination))
                {
                    throw new IOException("The destination appeared while Kaor was being installed: " + destination);
                }
                if (attempt < attempts)
                {
                    Thread.Sleep(100 * attempt);
                }
            }
            throw new IOException("Could not move the verified Kaor directory into place.", lastError);
        }

        private static string ToHex(byte[] bytes)
        {
            return BitConverter.ToString(bytes).Replace("-", "").ToLowerInvariant();
        }

        private static bool FixedEquals(string left, string right)
        {
            if (left == null || right == null || left.Length != right.Length)
            {
                return false;
            }
            var difference = 0;
            for (var index = 0; index < left.Length; index++)
            {
                difference |= Char.ToLowerInvariant(left[index]) ^ Char.ToLowerInvariant(right[index]);
            }
            return difference == 0;
        }

        private static void ThrowIfCancelled(Func<bool> cancelled)
        {
            if (cancelled != null && cancelled())
            {
                throw new OperationCanceledException();
            }
        }

        private static string FormatBytes(long value)
        {
            return String.Format("{0:N1} GiB", value / 1073741824.0);
        }

        private static void DeleteGeneratedFile(string path)
        {
            if (String.IsNullOrEmpty(path))
            {
                return;
            }
            Exception lastError = null;
            const int attempts = 8;
            for (var attempt = 1; attempt <= attempts; attempt++)
            {
                try
                {
                    if (!File.Exists(path))
                    {
                        return;
                    }
                    File.Delete(path);
                    return;
                }
                catch (IOException error)
                {
                    lastError = error;
                }
                catch (UnauthorizedAccessException error)
                {
                    lastError = error;
                }
                if (attempt < attempts)
                {
                    Thread.Sleep(100 * attempt);
                }
            }
            throw new IOException("Could not remove temporary release file: " + path, lastError);
        }

        private static void DeleteGeneratedDirectory(string path)
        {
            if (String.IsNullOrEmpty(path))
            {
                return;
            }
            Exception lastError = null;
            const int attempts = 8;
            for (var attempt = 1; attempt <= attempts; attempt++)
            {
                try
                {
                    if (!Directory.Exists(path))
                    {
                        return;
                    }
                    Directory.Delete(path, true);
                    return;
                }
                catch (IOException error)
                {
                    lastError = error;
                }
                catch (UnauthorizedAccessException error)
                {
                    lastError = error;
                }
                if (attempt < attempts)
                {
                    Thread.Sleep(100 * attempt);
                }
            }
            throw new IOException("Could not remove temporary extraction directory: " + path, lastError);
        }

        private static void TryDeleteGeneratedFile(string path)
        {
            try
            {
                DeleteGeneratedFile(path);
            }
            catch
            {
                // Preserve the original assembly, validation, extraction, or cancellation error.
            }
        }

        private static void TryDeleteGeneratedDirectory(string path)
        {
            try
            {
                DeleteGeneratedDirectory(path);
            }
            catch
            {
                // Preserve the original assembly, validation, extraction, or cancellation error.
            }
        }
    }

    internal sealed class InstallerForm : Form
    {
        private readonly string manifestPath;
        private readonly Label statusLabel;
        private readonly ProgressBar progressBar;
        private readonly Button cancelButton;
        private readonly BackgroundWorker worker;
        private InstallResult result;

        public int ExitCode { get; private set; }

        public InstallerForm(string manifestPath)
        {
            this.manifestPath = manifestPath;
            Text = "Kaor NVIDIA Release Installer";
            ClientSize = new Size(520, 150);
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            MinimizeBox = true;
            StartPosition = FormStartPosition.CenterScreen;

            statusLabel = new Label
            {
                AutoEllipsis = true,
                Location = new Point(22, 22),
                Size = new Size(476, 38),
                Text = "Preparing release verification..."
            };
            progressBar = new ProgressBar
            {
                Location = new Point(22, 65),
                Size = new Size(476, 22),
                Minimum = 0,
                Maximum = 100
            };
            cancelButton = new Button
            {
                Location = new Point(398, 103),
                Size = new Size(100, 30),
                Text = "Cancel"
            };
            cancelButton.Click += delegate { worker.CancelAsync(); cancelButton.Enabled = false; };
            Controls.Add(statusLabel);
            Controls.Add(progressBar);
            Controls.Add(cancelButton);

            worker = new BackgroundWorker
            {
                WorkerReportsProgress = true,
                WorkerSupportsCancellation = true
            };
            worker.DoWork += DoInstall;
            worker.ProgressChanged += delegate(object sender, ProgressChangedEventArgs args)
            {
                progressBar.Value = Math.Max(0, Math.Min(100, args.ProgressPercentage));
                statusLabel.Text = Convert.ToString(args.UserState);
            };
            worker.RunWorkerCompleted += InstallCompleted;
            Shown += delegate { worker.RunWorkerAsync(); };
            FormClosing += delegate(object sender, FormClosingEventArgs args)
            {
                if (worker.IsBusy)
                {
                    worker.CancelAsync();
                    args.Cancel = true;
                }
            };
        }

        private void DoInstall(object sender, DoWorkEventArgs args)
        {
            try
            {
                var destination = Path.GetDirectoryName(Path.GetFullPath(manifestPath));
                result = ReleaseInstaller.Install(
                    manifestPath,
                    destination,
                    delegate(int value, string message) { worker.ReportProgress(value, message); },
                    delegate { return worker.CancellationPending; });
            }
            catch (OperationCanceledException)
            {
                args.Cancel = true;
            }
        }

        private void InstallCompleted(object sender, RunWorkerCompletedEventArgs args)
        {
            cancelButton.Enabled = false;
            if (args.Cancelled)
            {
                MessageBox.Show(this, "Installation cancelled.", "Kaor", MessageBoxButtons.OK, MessageBoxIcon.Information);
                Close();
                return;
            }
            if (args.Error != null)
            {
                ExitCode = 1;
                Program.WriteErrorLog(args.Error);
                MessageBox.Show(this, args.Error.Message, "Kaor release verification failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Close();
                return;
            }

            var launch = MessageBox.Show(
                this,
                "Kaor was verified and extracted successfully. Start it now?",
                "Kaor",
                MessageBoxButtons.YesNo,
                MessageBoxIcon.Information);
            if (launch == DialogResult.Yes && result != null)
            {
                Process.Start(Path.Combine(result.TargetDirectory, "Kaor.exe"));
            }
            Close();
        }
    }

    internal static class Program
    {
        private static string ErrorLogPath()
        {
            var executable = Application.ExecutablePath;
            return Path.Combine(
                Path.GetDirectoryName(executable),
                Path.GetFileNameWithoutExtension(executable) + ".error.log");
        }

        internal static void WriteErrorLog(Exception error)
        {
            try
            {
                File.WriteAllText(ErrorLogPath(), error.ToString(), new UTF8Encoding(false));
            }
            catch
            {
                // The original installer error remains visible in the GUI or stderr.
            }
        }

        private static void ClearErrorLog()
        {
            try
            {
                if (File.Exists(ErrorLogPath()))
                {
                    File.Delete(ErrorLogPath());
                }
            }
            catch
            {
                // A stale log must not prevent verification or installation.
            }
        }

        [STAThread]
        private static int Main(string[] args)
        {
            try
            {
                ClearErrorLog();
                if (args.Length == 2 && args[0] == "--verify-only")
                {
                    ReleaseInstaller.VerifyOnly(args[1], delegate { }, delegate { return false; });
                    return 0;
                }
                if (args.Length == 3 && args[0] == "--headless")
                {
                    ReleaseInstaller.Install(args[1], args[2], delegate { }, delegate { return false; });
                    return 0;
                }
                if (args.Length != 0)
                {
                    throw new ArgumentException(
                        "Usage: --verify-only MANIFEST or --headless MANIFEST DESTINATION_BASE");
                }

                var executable = Application.ExecutablePath;
                var prefix = Path.GetFileNameWithoutExtension(executable);
                if (prefix.EndsWith("-Setup", StringComparison.OrdinalIgnoreCase))
                {
                    prefix = prefix.Substring(0, prefix.Length - "-Setup".Length);
                }
                var manifest = Path.Combine(Path.GetDirectoryName(executable), prefix + ".parts.json");
                Application.EnableVisualStyles();
                Application.SetCompatibleTextRenderingDefault(false);
                var form = new InstallerForm(manifest);
                Application.Run(form);
                return form.ExitCode;
            }
            catch (Exception error)
            {
                Console.Error.WriteLine(error.ToString());
                WriteErrorLog(error);
                if (args.Length == 0)
                {
                    MessageBox.Show(error.Message, "Kaor release verification failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                return 1;
            }
        }
    }
}
